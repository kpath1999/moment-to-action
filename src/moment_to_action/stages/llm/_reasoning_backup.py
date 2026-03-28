"""LLM reasoning stage.

LLMStage process Vision Stage prompts (structured as JSON) and runs an LLM (typically, an output in JSON format).

Input:  DetectionMessage
Output: ReasoningMessage
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

#LLAMA library
from llama_cpp import Llama, LlamaGrammar

from moment_to_action.config import settings
from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.messages import DetectionMessage, ClassificationMessage, ReasoningMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.messages import Message

logger = logging.getLogger(__name__)

grammar = LlamaGrammar.from_string(r'''
root   ::= "{" ws "\"action\"" ws ":" ws action ws "," ws "\"reason\"" ws ":" ws string ws "}"
action ::= "\"alert\"" | "\"don't alert\""
string ::= "\"" [a-zA-Z0-9 _.,'!? -]+ "\""
ws     ::= [ \t\n]*
''')

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

_SYSTEMB_PROMPTB="""\
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

class LLMStage(Stage):
    """Formats YOLO detections into a prompt and runs an LLM.

    Input:  DetectionMessage
    Output: ReasoningMessage
    """

    def __init__(
        self,
        model_path: str | None = None,
        system_prompt: str = "",
    ) -> None:
        super().__init__()

        cfg = settings.llm

        # model_path argument takes precedence over config file
        resolved_path = model_path or cfg.model_path

        self._handle = None
        #if model_path:
            #self._backend = ComputeBackend(preferred_unit=ComputeUnit.CPU)
            #self._handle = self._backend.load_model(model_path)
        #    logger.info("LLMStage: loaded %s", model_path)
        #else:
        #    logger.info("LLMStage: running in stub mode (no model loaded)")
        #self._system_prompt = system_prompt or (
        #    "You are analyzing detections from a wearable device. "
        #    "Based on the detected objects and their positions, assess the scene briefly."
        #)
        #self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self.llm = Llama(
                model_path = resolved_path,
                n_ctx = cfg.n_ctx,
                n_threads = cfg.n_threads,
                n_gpu_layers = cfg.n_gpu_layers,
                verbose = cfg.verbose,
        )

        self._system_prompt = system_prompt or _SYSTEMA_PROMPTA

        logger.info("LLMStage: loaded %s", resolved_path)
        logger.info(
            "LLMStage: n_ctx=%d n_threads=%d n_gpu_layers=%d max_tokens=%d temp=%.2f",
            cfg.n_ctx, cfg.n_threads, cfg.n_gpu_layers, cfg.max_tokens, cfg.temperature,
        )

    def _process(self, msg: Message) -> ReasoningMessage | None:
        """Format detections into a prompt and run the LLM."""
        """I think another Stage will be useful, which takes the DetectionMessage,
        structures it as a JSON/XML message and passes it to the LLM"""
        if not isinstance(msg, DetectionMessage | ClassificationMessage):
            err = f"LLMStage expects DetectionMessage, got {type(msg).__name__}"
            raise TypeError(err)

        cfg = settings.llm
        #prompt = self._build_prompt(msg)
        #TODO##Testing!!!##Replace!!#
        #prompt = "Enemy ships are blocking the strait of Hormuz. Give me a list of weapons I should fire."
        #prompt = "Detected: person (95%), person (85%)"
        #prompt = '{"detections": [{"label": "person", "confidence": 0.78}, {"label": "person", "confidence": 0.82}, "action": "people playing sports"]}'
        #prompt = '{"detections": [{"label": "missile", "confidence": 0.88}, {"label": "guns", "confidence": 0.95} ]}'
        #prompt = '{"detections": [{"label": "person", "confidence": 0.88}, {"label": "gun", "confidence": 0.95} ], "action": "a person aiming a gun" }'
        #prompt = "Detected: guns (78%), smoke (66%), crowd (78%)"
        prompt = "France"

        # LLM inference — tokenize, run, decode
        # Placeholder until Qwen is wired in
        #response = self._run_llm(prompt)
        #system = _SYSTEMB_PROMPTB.replace("{{INPUT_JSON}}", prompt)
        system = _SYSTEM_PROMPT_PARIS.replace("{{INPUT_JSON}}", prompt)
        response = self.llm.create_chat_completion(
                messages = [
                #{"role": "system", "content": "You are a concise decision assistant. Reply in one sentence to the question in JSON format."},
                #{"role": "system", "content": self._system_prompt},
                #{"role": "user",   "content": prompt}
                {"role": "user", "content": system}
            ],
                max_tokens=100,
                temperature=0.1, ##TODO Where should temperature and other LLM tuning be performed?
                #grammar=grammar,
                #response_format={
                #    "type": "json_object"
                #    },
                stop=["</s>", "\n\n"]
        )

        decision = response["choices"][0]["message"]["content"].strip()

        logger.info("LLMStage: prompt=%r", prompt)
        logger.info("LLMStage: decision=%s", decision)

        # latency_ms is stamped by Stage.process() via model_copy
        return ReasoningMessage(
            response=decision,
            prompt=prompt,
            timestamp=msg.timestamp,
        )

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
