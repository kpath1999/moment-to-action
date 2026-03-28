"""LLM reasoning stage.

LLMStage processes Vision Stage prompts (structured as JSON) and runs an LLM.

Input:  DetectionMessage | ClassificationMessage
Output: ReasoningMessage
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llama_cpp import Llama, LlamaGrammar

from moment_to_action.config import settings
from moment_to_action.messages import DetectionMessage, ClassificationMessage, ReasoningMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.messages import Message

logger = logging.getLogger(__name__)

# ── Grammar ─────────────────────────────────────────────────────────────────
# Enforces valid JSON output — model cannot produce anything else.
grammar = LlamaGrammar.from_string(r'''
root        ::= "{" ws "\"decision\"" ws ":" ws decision ws ","
                    ws "\"confidence\"" ws ":" ws number ws ","
                    ws "\"risk_factors\"" ws ":" ws risk_array ws ","
                    ws "\"reasoning\"" ws ":" ws string ws "}"
decision    ::= "\"alert\"" | "\"no_alert\""
risk_array  ::= "[" ws (string (ws "," ws string)*)? ws "]"
number      ::= [0-9] "." [0-9]+
string      ::= "\"" [a-zA-Z0-9 _.,'!?/:-]+ "\""
ws          ::= [ \t\n]*
''')

# ── System prompt ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a safety monitoring system on an edge wearable device.

Analyze structured object detections and decide whether to trigger an alert.

## RULES
- Output MUST be valid JSON only. No extra text.
- Be conservative: if uncertain, choose "alert".
- Base your decision ONLY on provided detections.

## ALERT CRITERIA
Trigger "alert" if ANY of the following:
- A weapon (knife, gun, bat) is present near a person
- Physical contact between people is detected
- Confidence of dangerous object > 0.7 AND a person is present

Otherwise return "no_alert".

## OUTPUT FORMAT
{
  "decision": "alert" | "no_alert",
  "confidence": 0.0-1.0,
  "risk_factors": ["factor1", ...],
  "reasoning": "one short sentence"
}"""

# ── Metrics dataclass ────────────────────────────────────────────────────────
@dataclass
class InferenceMetrics:
    turn:               int
    input_tokens:       int   = 0
    output_tokens:      int   = 0
    wall_time_s:        float = 0.0
    prefill_ms:         float = 0.0
    decode_ms:          float = 0.0
    prefill_tps:        float = 0.0
    decode_tps:         float = 0.0
    kv_cache_tokens:    int   = 0

    def log(self) -> None:
        logger.info(
            "metrics | turn=%d in_tok=%d out_tok=%d wall=%.2fs "
            "prefill=%.0fms(%.1f t/s) decode=%.0fms(%.1f t/s) kv=%d",
            self.turn, self.input_tokens, self.output_tokens, self.wall_time_s,
            self.prefill_ms, self.prefill_tps,
            self.decode_ms, self.decode_tps,
            self.kv_cache_tokens,
        )

    def pretty(self) -> str:
        return (
            f"\n── Turn {self.turn} metrics ──────────────────────\n"
            f"  input_tokens      {self.input_tokens}\n"
            f"  output_tokens     {self.output_tokens}\n"
            f"  wall_time_s       {self.wall_time_s:.3f}\n"
            f"  prefill_ms        {self.prefill_ms:.1f}  ({self.prefill_tps:.1f} t/s)\n"
            f"  decode_ms         {self.decode_ms:.1f}  ({self.decode_tps:.1f} t/s)\n"
            f"  kv_cache_tokens   {self.kv_cache_tokens}\n"
        )


# ── Stage ────────────────────────────────────────────────────────────────────
class LLMStage(Stage):
    """Formats YOLO/CLIP detections into a scene graph prompt and runs an LLM.

    Input:  DetectionMessage | ClassificationMessage
    Output: ReasoningMessage
    """

    def __init__(
        self,
        model_path: str | None = None,
        system_prompt: str = "",
        max_history_turns: int = 3,
    ) -> None:
        super().__init__()

        cfg = settings.llm
        resolved_path = model_path or cfg.model_path

        self.llm = Llama(
            model_path=resolved_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=cfg.verbose,
        )

        self._system_prompt = system_prompt or _SYSTEM_PROMPT

        # sliding window — each turn appends 2 items (user + assistant)
        self._history: deque[dict] = deque(maxlen=max_history_turns * 2)
        self._turn: int = 0

        logger.info("LLMStage: loaded %s", resolved_path)
        logger.info(
            "LLMStage: n_ctx=%d n_threads=%d n_gpu_layers=%d "
            "max_tokens=%d temp=%.2f history_turns=%d",
            cfg.n_ctx, cfg.n_threads, cfg.n_gpu_layers,
            cfg.max_tokens, cfg.temperature, max_history_turns,
        )

    # ── internal helpers ─────────────────────────────────────────────────────

    def _build_messages(self, user_content: str) -> list[dict]:
        """Assemble [system] + sliding history window + new user message."""
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._history)           # already bounded by deque maxlen
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_scene_graph(self, msg: DetectionMessage | ClassificationMessage) -> str:
        """Convert detection message into a structured scene graph string."""
        if isinstance(msg, DetectionMessage):
            detections = [
                {"label": box.label, "confidence": round(box.confidence, 2),
                 "bbox": [round(box.x1), round(box.y1), round(box.x2), round(box.y2)]}
                for box in msg.top(5)
            ]
        else:
            # ClassificationMessage — adapt field names as needed
            detections = [
                {"label": cls.label, "confidence": round(cls.confidence, 2)}
                for cls in msg.top(5)
            ]

        return json.dumps({"detections": detections}, indent=2)

    def _extract_metrics(
        self,
        response: dict,
        t_start: float,
        t_end: float,
        messages: list[dict],
    ) -> InferenceMetrics:
        """Pull timing data from llama-cpp-python response + wall clock."""
        usage         = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        comp_tokens   = usage.get("completion_tokens", 0)
        wall          = t_end - t_start

        # llama.cpp internal timings (available in most builds)
        timings = None
        try:
            timings = self.llm._ctx.get_timings()
        except Exception:
            pass

        if timings and timings.t_eval_ms > 0 and timings.t_p_eval_ms > 0:
            prefill_ms  = timings.t_p_eval_ms
            decode_ms   = timings.t_eval_ms
            prefill_tps = timings.n_p_eval / (prefill_ms / 1000) if prefill_ms else 0
            decode_tps  = timings.n_eval   / (decode_ms  / 1000) if decode_ms  else 0
        else:
            # fallback estimates from wall clock
            prefill_ms  = 0.0
            decode_ms   = wall * 1000
            prefill_tps = 0.0
            decode_tps  = comp_tokens / wall if wall > 0 else 0.0

        return InferenceMetrics(
            turn            = self._turn,
            input_tokens    = prompt_tokens,
            output_tokens   = comp_tokens,
            wall_time_s     = round(wall, 3),
            prefill_ms      = round(prefill_ms, 1),
            decode_ms       = round(decode_ms,  1),
            prefill_tps     = round(prefill_tps, 1),
            decode_tps      = round(decode_tps,  1),
            kv_cache_tokens = prompt_tokens,
        )

    # ── main process ─────────────────────────────────────────────────────────

    def _process(self, msg: Message) -> ReasoningMessage | None:
        if not isinstance(msg, (DetectionMessage, ClassificationMessage)):
            raise TypeError(
                f"LLMStage expects DetectionMessage or ClassificationMessage, "
                f"got {type(msg).__name__}"
            )

        cfg = settings.llm
        self._turn += 1

        # build prompt
        scene_graph = self._build_scene_graph(msg)
        messages    = self._build_messages(scene_graph)

        # inference
        t_start = time.perf_counter()
        response = self.llm.create_chat_completion(
            messages      = messages,
            max_tokens    = cfg.max_tokens,
            temperature   = cfg.temperature,
            top_k         = 20,
            top_p         = 0.8,
            repeat_penalty= 1.5,
            grammar       = grammar,
            stop          = ["</s>", "\n\n"],
        )
        t_end = time.perf_counter()

        decision = response["choices"][0]["message"]["content"].strip()

        # metrics
        metrics = self._extract_metrics(response, t_start, t_end, messages)
        metrics.log()
        print(metrics.pretty())     # visible during testing

        # update sliding window history
        self._history.append({"role": "user",      "content": scene_graph})
        self._history.append({"role": "assistant",  "content": decision})

        logger.info("LLMStage | turn=%d decision=%s", self._turn, decision)

        return ReasoningMessage(
            response  = decision,
            prompt    = scene_graph,
            timestamp = msg.timestamp,
        )

    def reset_history(self) -> None:
        """Clear conversation history (call between sessions)."""
        self._history.clear()
        self._turn = 0
        logger.info("LLMStage: history reset")


# ── Standalone test loop ─────────────────────────────────────────────────────
# Run directly:  python _reasoning.py
# This tests the LLM stage in isolation without the full pipeline.

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # ── preset scene graphs ──────────────────────────────────────────────────
    # Replace model_path with your actual path
    MODEL_PATH = "/home/ubuntu/moment-to-action/llm_models/qwen2.5-1.5b-instruct-q5_k_m.gguf"

    PRESET_SCENES = [
        # turn 1 — clear threat
        '{"detections": [{"label": "person", "confidence": 0.91}, '
        '{"label": "knife", "confidence": 0.83}]}',

        # turn 2 — same scene, knife more prominent (tests KV cache reuse)
        '{"detections": [{"label": "person", "confidence": 0.89}, '
        '{"label": "knife", "confidence": 0.91}, '
        '{"label": "blood", "confidence": 0.72}]}',

        # turn 3 — threat resolved
        '{"detections": [{"label": "person", "confidence": 0.88}]}',

        # turn 4 — new threat (tests sliding window)
        '{"detections": [{"label": "person", "confidence": 0.95}, '
        '{"label": "person", "confidence": 0.87}, '
        '{"label": "physical-contact", "confidence": 0.79}]}',

        # turn 5 — benign
        '{"detections": [{"label": "person", "confidence": 0.92}, '
        '{"label": "dog", "confidence": 0.88}]}',
    ]

    # ── init stage directly (bypasses pipeline machinery) ───────────────────
    # We instantiate Llama directly here to avoid needing settings config.
    print(f"\nLoading model: {MODEL_PATH}")
    print("=" * 60)

    llm_handle = Llama(
        model_path  = MODEL_PATH,
        n_ctx       = 2048,
        n_threads   = 4,
        n_gpu_layers= 0,
        verbose     = False,
    )

    history: deque[dict] = deque(maxlen=6)   # 3 turns × 2
    turn = 0

    def run_turn(scene: str) -> None:
        global turn
        turn += 1

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": scene})

        t0 = time.perf_counter()
        response = llm_handle.create_chat_completion(
            messages       = messages,
            max_tokens     = 80,
            temperature    = 0.1,
            top_k          = 20,
            top_p          = 0.8,
            repeat_penalty = 1.5,
            grammar        = grammar,
            stop           = ["</s>", "\n\n"],
        )
        t1 = time.perf_counter()

        usage     = response.get("usage", {})
        in_tok    = usage.get("prompt_tokens", 0)
        out_tok   = usage.get("completion_tokens", 0)
        wall      = t1 - t0
        decision  = response["choices"][0]["message"]["content"].strip()

        # internal timings
        try:
            tm        = llm_handle._ctx.get_timings()
            prefill_ms= round(tm.t_p_eval_ms, 1)
            decode_ms = round(tm.t_eval_ms,   1)
            p_tps     = round(tm.n_p_eval / (tm.t_p_eval_ms / 1000), 1) if tm.t_p_eval_ms else 0
            d_tps     = round(tm.n_eval   / (tm.t_eval_ms   / 1000), 1) if tm.t_eval_ms   else 0
        except Exception:
            prefill_ms= 0.0
            decode_ms = round(wall * 1000, 1)
            p_tps     = 0.0
            d_tps     = round(out_tok / wall, 1) if wall else 0

        print(f"\n── Turn {turn} ──────────────────────────────────────────")
        print(f"  input:          {scene[:80]}...")
        print(f"  decision:       {decision}")
        print(f"  input_tokens:   {in_tok}")
        print(f"  output_tokens:  {out_tok}")
        print(f"  wall_time:      {wall:.3f}s")
        print(f"  prefill:        {prefill_ms}ms  @ {p_tps} t/s")
        print(f"  decode:         {decode_ms}ms  @ {d_tps} t/s")
        print(f"  kv_cache_size:  {in_tok} tokens")

        history.append({"role": "user",      "content": scene})
        history.append({"role": "assistant", "content": decision})

    # ── run all preset scenes ────────────────────────────────────────────────
    print(f"\nRunning {len(PRESET_SCENES)} preset scenes...\n")
    for scene in PRESET_SCENES:
        run_turn(scene)

    print("\n" + "=" * 60)
    print("Test complete.")
    print("Key things to observe:")
    print("  - Turn 1 prefill is highest (system prompt + scene)")
    print("  - Turn 2-3 prefill grows slightly (history accumulates)")
    print("  - Turn 4-5 prefill stabilises (sliding window bounded)")
    print("  - Decode time should stay roughly constant across turns")
