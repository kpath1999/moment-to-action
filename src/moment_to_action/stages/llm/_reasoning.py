"""LLM reasoning stage.

LLMStage process Vision Stage prompts (structured as JSON) and runs an LLM.

Input:  DetectionMessage
Output: ReasoningMessage
"""

from __future__ import annotations

import logging

# time
# deque
from collections import deque

# dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

# httpx server for metrics
import httpx
import psutil

# OpenAI library for HTTP communication
from openai import OpenAI

from moment_to_action.config.slm_config.slm_config import settings
from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.messages import ClassificationMessage, DetectionMessage, ReasoningMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.models import ModelID, ModelManager

logger = logging.getLogger(__name__)

"""
grammar = LlamaGrammar.from_string(r'''
root   ::= "{" ws "\"action\"" ws ":" ws action ws "," ws "\"reason\"" ws ":" ws string ws "}"
action ::= "\"alert\"" | "\"don't alert\""
string ::= "\"" [a-zA-Z0-9 _.,'!? -]+ "\""
ws     ::= [ \t\n]*
''')
"""

# Few-shot system prompt — examples teach the model what each field means
# and how to reason from detections to a decision.
_SYSTEM_PROMPT = """\
You analyze object detections from a wearable camera and decide what action to take.

Respond only in JSON with exactly these fields:
- action: either "alert" or "don't alert"
- reason: one short sentence explaining your decision

Examples:

Detected: person (94%), dog (87%)
{"action": "don't alert", "reason": "people and animals present no immediate threat, so no need to alert"}

Detected: person (91%), knife (82%)
{"action": "alert", "reason": "potential weapon detected near person, so need to alert"}

Detected: person (88%), person (76%)
{"action": "don't alert", "reason": "two people present no signs of danger, so no need to alert"}

Detected: person (93%), person (85%), physical-contact (79%)
{"action": "alert", "reason": "physical contact between people detected which is dangerous, so need to alert"}

Now respond the same way for the detections you receive.\
"""

_SYSTEM_PROMPT_PARIS = """\
What is the capital of {{INPUT_JSON}}\
"""

_SYSTEMA_PROMPTA = """\
You are a safety monitoring system running on an edge device.

Your task is to decide whether to trigger an alert based ONLY on structured detections.

## RULES

* Output MUST be valid JSON only. No extra text.
* Be conservative: if uncertain, choose "alert".
* Base your decision ONLY on provided data.
* Ignore anything not present in the input.
* Do NOT explain outside the JSON.

## ALERT CRITERIA

Trigger "alert" if ANY of the following are true:

* A weapon (knife, gun, bat, etc.) is present near a person
* A dangerous interaction is detected (e.g., person holding weapon)
* Confidence of dangerous object > 0.7 AND a person is present
* Multiple risk factors appear together

Otherwise return "no_alert".

## EXAMPLES

Input:
{
"detections": [
{"label": "person", "confidence": 0.95},
{"label": "knife", "confidence": 0.90}
]
}

Output:
{
"decision": "alert",
"confidence": 0.92,
"risk_factors": ["person_with_weapon"],
"reasoning": "Knife detected with high confidence near a person"
}

Input:
{
"detections": [
{"label": "dog", "confidence": 0.88}
]
}

Output:
{
"decision": "no_alert",
"confidence": 0.95,
"risk_factors": [],
"reasoning": "No dangerous objects or interactions detected"
}

Otherwise return "no_alert".

## INPUT

{{INPUT_JSON}}

As your output, you will tell me if there is an alert required or not, and a line of reasoning.\
"""

_SYSTEMB_PROMPTB = """\
You analyze object detections from a wearable camera and decide what action to take.

Respond only in JSON with exactly these fields:
- action: either "alert" or "don't alert" based on whether there is a threat or not
- reason: one short sentence explaining your decision\

## EXAMPLES

Input:
{
"detections": [
{"label": "person", "confidence": 0.95},
{"label": "knife", "confidence": 0.90}
],
"action": "a person holding a knife"
}

Output:
{
"decision": "alert",
"risk_factors": ["person_with_weapon"],
"reasoning": "The person is holding a knife, so there is a threat as he may attack."
}

Now respond the same way for following detections:

{{INPUT_JSON}}\
"""


# ── Metrics dataclass ────────────────────────────────────────────────────────
@dataclass
class InferenceMetrics:
    turn: int
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    kv_cache_tokens: int = 0

    def log(self) -> None:
        logger.info(
            "metrics | turn=%d in_tok=%d out_tok=%d wall=%.2fs "
            "prefill=%.0fms(%.1f t/s) decode=%.0fms(%.1f t/s) kv=%d",
            self.turn,
            self.input_tokens,
            self.output_tokens,
            self.wall_time_s,
            self.prefill_ms,
            self.prefill_tps,
            self.decode_ms,
            self.decode_tps,
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


class LLMStage(Stage):
    """Formats YOLO detections into a prompt and runs an LLM.

    Input:  DetectionMessage
    Output: ReasoningMessage
    """

    _LLAMA_BASE_URL = "http://localhost:8080"

    _backend: ComputeBackend | None
    _handle: object | None

    def __init__(
        self,
        model_id: ModelID | None = None,
        system_prompt: str = "",
        manager: ModelManager | None = None,
        max_history_turns: int = 3,
    ) -> None:
        super().__init__()

        cfg = settings.llm

        # model_path argument takes precedence over config file

        self._handle = None
        if model_id is not None:
            # Resolve model path through the manager — downloads/caches as needed.
            if manager is None:
                msg = "Model manager is required when a model ID is provided!"
                raise ValueError(msg)

            model_path = manager.get_path(model_id)
            self._backend = ComputeBackend(preferred_unit=ComputeUnit.CPU)
            logger.info("ReasoningStage: loaded %s", model_path)
        else:
            self._backend = None
            logger.info("ReasoningStage: running in stub mode (no model loaded)")

        resolved_path = model_path or cfg.model_path

        self._system_prompt = system_prompt or _SYSTEMA_PROMPTA

        self._history: deque[dict] = deque(maxlen=max_history_turns * 2)
        self._turn: int = 0

        logger.info("LLMStage: loaded %s", resolved_path)
        logger.info(
            "LLMStage: n_ctx=%d n_threads=%d n_gpu_layers=%d max_tokens=%d temp=%.2f",
            cfg.n_ctx,
            cfg.n_threads,
            cfg.n_gpu_layers,
            cfg.max_tokens,
            cfg.temperature,
        )

        logger.info(
            "LLMStage: n_ctx=%d n_threads=%d n_gpu_layers=%d "
            "max_tokens=%d temp=%.2f history_turns=%d",
            cfg.n_ctx,
            cfg.n_threads,
            cfg.n_gpu_layers,
            cfg.max_tokens,
            cfg.temperature,
            max_history_turns,
        )

    def _build_messages(self, user_content: str) -> list[dict]:
        """Assemble [system] + sliding history window + new user message."""
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._history)  # already bounded by deque maxlen
        messages.append({"role": "user", "content": user_content})
        return messages

    """
    def _extract_metrics(
        self,
        response: dict,
        t_start: float,
        t_end: float,
        messages: list[dict],
    ) -> InferenceMetrics:
        #Pull timing data from llama-cpp-python response + wall clock.
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        comp_tokens = usage.get("completion_tokens", 0)
        wall = t_end - t_start

        # llama.cpp internal timings (available in most builds)
        timings = None
        try:
            timings = self.llm._ctx.get_timings()
        except Exception:
            pass

        if timings and timings.t_eval_ms > 0 and timings.t_p_eval_ms > 0:
            prefill_ms = timings.t_p_eval_ms
            decode_ms = timings.t_eval_ms
            prefill_tps = timings.n_p_eval / (prefill_ms / 1000) if prefill_ms else 0
            decode_tps = timings.n_eval / (decode_ms / 1000) if decode_ms else 0
        else:
            # fallback estimates from wall clock
            prefill_ms = 0.0
            decode_ms = wall * 1000
            prefill_tps = 0.0
            decode_tps = comp_tokens / wall if wall > 0 else 0.0

        return InferenceMetrics(
            turn=self._turn,
            input_tokens=prompt_tokens,
            output_tokens=comp_tokens,
            wall_time_s=round(wall, 3),
            prefill_ms=round(prefill_ms, 1),
            decode_ms=round(decode_ms, 1),
            prefill_tps=round(prefill_tps, 1),
            decode_tps=round(decode_tps, 1),
            kv_cache_tokens=prompt_tokens,
        )
    """

    def _process(self, msg: Message) -> ReasoningMessage | None:
        """Format detections into a prompt and run the LLM."""
        """I think another Stage will be useful, which takes the DetectionMessage,
        structures it as a JSON/XML message and passes it to the LLM"""
        if not isinstance(msg, DetectionMessage | ClassificationMessage):
            err = f"LLMStage expects DetectionMessage, got {type(msg).__name__}"
            raise TypeError(err)

        self._turn += 1
        # TODO##Testing!!!##Replace!!#
        # prompt = "Detected: person (95%), person (85%)"
        # prompt = '{"detections": [{"label": "person", "confidence": 0.78}, {"label": "person", "confidence": 0.82}, "action": "people playing sports"]}'
        # prompt = '{"detections": [{"label": "missile", "confidence": 0.88}, {"label": "guns", "confidence": 0.95} ]}'
        prompt = '{"detections": [{"label": "person", "confidence": 0.88}, {"label": "gun", "confidence": 0.95} ], "action": "a person aiming a gun" }'
        # prompt = "Detected: guns (78%), smoke (66%), crowd (78%)"
        # prompt = "France"

        # LLM inference — tokenize, run, decode
        system = _SYSTEMB_PROMPTB.replace("{{INPUT_JSON}}", prompt)
        # system = _SYSTEM_PROMPT_PARIS.replace("{{INPUT_JSON}}", prompt)
        messages = [{"role": "user", "content": system}]

        # Start client
        client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

        # response = self.llm.create_chat_completion(
        response = client.chat.completions.create(
            model="any",
            messages=messages,
            max_tokens=100,
            temperature=0.1,  ##TODO Where should temperature and other LLM tuning be performed?
            # grammar=grammar,
            # response_format={
            #    "type": "json_object"
            #    },
            stop=["</s>", "\n\n"],
        )

        # decision = response["choices"][0]["message"]["content"].strip()
        decision = response.choices[0].message.content.strip()

        logger.info("LLMStage: prompt=%r", prompt)
        logger.info("LLMStage: decision=%s", decision)

        return ReasoningMessage(
            response=decision,
            prompt=prompt,
            timestamp=msg.timestamp,
        )

    def reset_history(self) -> None:
        """Clear conversation history (call between sessions)."""
        self._history.clear()
        self._turn = 0
        logger.info("LLMStage: history reset")

    def _build_prompt(self, msg: DetectionMessage) -> str:
        lines = [self._system_prompt, "", "Detections:"]
        lines.extend(
            f"  - {box.label} (confidence: {box.confidence:.2f}, "
            f"position: [{box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f}])"
            for box in msg.top(5)
        )
        lines.append("\nWhat is happening in this scene?")
        return "\n".join(lines)

    def _run_llm(self, prompt: str) -> str:
        # NOTE(kausar): integrate with Kausar's LLM arch. LLM is a stage that
        # ingests the message, performs inference dispatched via ComputeBackend.
        # For now return the prompt so the pipeline is runnable end-to-end.
        return f"[LLM stub] Received prompt with {len(prompt)} chars."

    def _server_rss_bytes(self) -> int:
        for proc in psutil.process_iter(["name", "memory_info"]):
            if "llama-server" in proc.info["name"]:
                return proc.info["memory_info"].rss
        return 0

    # in LLMStage
    # These log the metrics specific to the LLM, and send them to the logging method in _base.py
    def _llm_metrics(self) -> dict:
        slots = httpx.get(f"{self._LLAMA_BASE_URL}/slots").json()
        slot = slots[0]
        return {
            "prompt_ms": 0.0,
            # "gen_ms": self._last_latency_ms,       # stored in _process()
            "gen_ms": 0.0,  # stored in _process()
            # "prompt_tokens": self._last_usage.prompt_tokens,
            # "gen_tokens": self._last_usage.completion_tokens,
            "kv_cache_used": slot.get("n_past", 0),
            "kv_cache_total": slot.get("n_ctx", 512),
            "server_rss_bytes": self._server_rss_bytes(),
        }
