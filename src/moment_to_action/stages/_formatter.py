"""Prompt formatter stage.

PromptFormatterStage sits between vision stages (YOLO, MobileCLIP, …) and
LLMStage. It converts any upstream message into a PromptMessage containing
a fully-formed LLM prompt.

Design
------
The idea is to handle each message type differently to format them into
something the LLM can use. Different models using the same message type
will be covered.
* A **handler registry** maps ``type[Message] -> FormatHandler`` so new
  upstream stages can be supported by registering one function — no changes
  to this file needed.
* Built-in handlers are registered automatically for ``DetectionMessage`` and
  ``ClassificationMessage``. Pass ``extra_handlers`` at construction time to
  override or extend.
* Optional **confidence filtering** and **top-K limiting** are applied inside
  each built-in handler so the LLM never sees noisy low-confidence detections.
* The final prompt is rendered via a **PromptTemplate** — either JSON-style
  (default) or plain natural-language, and fully replaceable.

Input:  DetectionMessage | ClassificationMessage | any registered type
Output: PromptMessage
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

from moment_to_action.messages import ClassificationMessage, DetectionMessage
from moment_to_action.messages.prompt import PromptMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for a handler function
# ---------------------------------------------------------------------------

# A handler receives the raw message plus filtering params and returns a
# structured dict that will be handed to the template renderer.
# Signature: (msg, min_confidence, top_k) -> dict
FormatHandler = Callable[["Message", float, int], dict[str, Any]]


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------


def _handle_detection(
    msg: DetectionMessage,
    min_confidence: float,
    top_k: int,
) -> dict:
    """Serialize a DetectionMessage into a structured context dict.

    Applies confidence filtering and top-K limiting so the prompt stays
    focused on the most salient detections.
    """
    boxes = [b for b in msg.boxes if b.confidence >= min_confidence]
    boxes = sorted(boxes, key=lambda b: -b.confidence)[:top_k]

    detections = [{"label": b.label, "confidence": round(b.confidence, 2)} for b in boxes]
    return {
        "source": "detection",
        "detections": detections,
    }


def _handle_classification(
    msg: ClassificationMessage,
    min_confidence: float,
    top_k: int,
) -> dict:
    """Serialize a ClassificationMessage into a structured context dict.

    Uses ``all_scores`` to apply confidence filtering and top-K limiting,
    giving the LLM a ranked view of the full distribution rather than just
    the single winning label.  The winning label/confidence from the message
    are also surfaced explicitly so the prompt is unambiguous.
    """
    # Filter and rank the full distribution
    filtered = {label: score for label, score in msg.all_scores.items() if score >= min_confidence}
    top_labels = sorted(filtered.items(), key=lambda kv: -kv[1])[:top_k]

    classifications = [
        {"label": label, "confidence": round(score, 2)} for label, score in top_labels
    ]

    return {
        "source": "classification",
        "winner": {"label": msg.label, "confidence": round(msg.confidence, 2)},
        "classifications": classifications,
    }


# ---------------------------------------------------------------------------
# Template protocol + built-in templates
# ---------------------------------------------------------------------------


class PromptTemplate(Protocol):
    """Callable that turns a context dict into a prompt string."""

    def __call__(self, context: dict) -> str: ...


def json_template(context: dict) -> str:
    """Render context as compact JSON — matches the existing system prompts."""
    return json.dumps(context, separators=(", ", ": "))


def natural_language_template(context: dict) -> str:
    """Render context as a plain English sentence list.

    Example output:
        Detected: person (0.95), knife (0.90)
        Classified: indoor scene (0.88)
    """
    parts = []

    if "detections" in context:
        items = ", ".join(f"{d['label']} ({d['confidence']:.0%})" for d in context["detections"])
        parts.append(f"Detected: {items}" if items else "Detected: nothing above threshold")

    if "classifications" in context:
        items = ", ".join(
            f"{c['label']} ({c['confidence']:.0%})" for c in context["classifications"]
        )
        parts.append(f"Classified: {items}" if items else "Classified: nothing above threshold")

    # Fall back gracefully for unknown context shapes from custom handlers
    if not parts:
        parts.append(json.dumps(context))

    return "\n".join(parts)


# Convenience map so callers can pass a string instead of a function
TEMPLATES: dict[str, PromptTemplate] = {
    "json": json_template,
    "natural": natural_language_template,
}


# ---------------------------------------------------------------------------
# PromptFormatterStage
# ---------------------------------------------------------------------------


class PromptFormatterStage(Stage):
    """Converts upstream vision messages into LLM-ready PromptMessages.

    Parameters
    ----------
    template:
        How to render the context dict into a prompt string.
        Pass ``"json"`` (default), ``"natural"``, or any callable with the
        signature ``(dict) -> str``.
    min_confidence:
        Detections/classifications below this threshold are dropped before
        the prompt is built.  Defaults to 0.0 (no additional filtering —
        relies on upstream stage thresholds).
    top_k:
        Maximum number of detections/labels to include in the prompt.
        The highest-confidence items are kept.  Defaults to 5.
    extra_handlers:
        Optional dict mapping ``type[Message]`` to a ``FormatHandler``
        callable.  Use this to override built-in handlers or add support
        for new upstream message types without touching this file::

            formatter = PromptFormatterStage(
                extra_handlers={
                    DepthMessage: my_depth_handler,
                }
            )

    Examples:
    --------
    Minimal usage with a YOLO pipeline::

        pipeline = Pipeline([
            YOLOStage(...),
            PromptFormatterStage(),          # JSON template, default filtering
            LLMStage(...),
        ])

    With MobileCLIP and natural-language prompts::

        pipeline = Pipeline([
            MobileCLIPStage(...),
            PromptFormatterStage(template="natural", min_confidence=0.6),
            LLMStage(...),
        ])

    Registering a custom handler for a new stage type::

        def my_pose_handler(msg, min_confidence, top_k):
            return {"pose_keypoints": msg.keypoints[:top_k]}

        formatter = PromptFormatterStage(
            extra_handlers={PoseMessage: my_pose_handler}
        )
    """

    def __init__(
        self,
        template: PromptTemplate | str = "json",
        min_confidence: float = 0.0,
        top_k: int = 5,
        extra_handlers: dict[type, FormatHandler] | None = None,
    ) -> None:
        super().__init__()

        # Resolve template
        if isinstance(template, str):
            if template not in TEMPLATES:
                msg = f"Unknown template {template!r}. Choose from: {list(TEMPLATES)}"
                raise ValueError(msg)
            self._template: PromptTemplate = TEMPLATES[template]
        else:
            self._template = template

        self._min_confidence = min_confidence
        self._top_k = top_k

        # Build the registry: built-ins first, then caller overrides
        self._registry: dict[type, FormatHandler] = {
            DetectionMessage: cast("FormatHandler", _handle_detection),
            ClassificationMessage: cast("FormatHandler", _handle_classification),
        }
        if extra_handlers:
            self._registry.update(extra_handlers)

        logger.info(
            "PromptFormatterStage: template=%s min_confidence=%.2f top_k=%d handlers=%s",
            getattr(self._template, "__name__", repr(self._template)),
            self._min_confidence,
            self._top_k,
            [t.__name__ for t in self._registry],
        )

    # ------------------------------------------------------------------
    # Public API — register handlers after construction
    # ------------------------------------------------------------------

    def register_handler(self, msg_type: type, handler: FormatHandler) -> None:
        """Register (or replace) a handler for a message type at runtime.

        Useful when building the pipeline dynamically or in tests::

            formatter.register_handler(AudioMessage, my_audio_handler)
        """
        self._registry[msg_type] = handler
        logger.debug("PromptFormatterStage: registered handler for %s", msg_type.__name__)

    # ------------------------------------------------------------------
    # Stage._process
    # ------------------------------------------------------------------

    def _process(self, msg: Message, _metrics: MetricsCollector) -> PromptMessage | None:
        """Look up the handler for this message type, format, and emit."""
        handler = self._registry.get(type(msg))

        if handler is None:
            msg_type = type(msg).__name__
            error_message = (
                "PromptFormatterStage: no handler registered for "
                f"{msg_type}. Register one via extra_handlers= or .register_handler()."
            )
            raise TypeError(error_message)

        # Build the structured context dict via the handler
        context = handler(msg, self._min_confidence, self._top_k)

        # Render to a prompt string via the template
        prompt = self._template(context)

        logger.debug(
            "PromptFormatterStage: source=%s items=%d prompt=%r",
            context.get("source", "unknown"),
            len(context.get("detections", context.get("classifications", []))),
            prompt[:120],
        )

        return PromptMessage(
            prompt=prompt,
            source_stage=type(msg).__name__,
            raw_context=context,
            timestamp=msg.timestamp,
        )
