"""Abstract Stage base class.

Stage is a lazy generator transformer: ``process()`` consumes an iterator of
messages and yields an iterator of messages. The base class owns the
cross-cutting concerns — input buffering/windowing, opening the per-item
``SpanType.STAGE`` span, and advancing the window — and delegates the pure
transform logic to ``_process(items) -> Iterator[Message]``.

MetricsCollector is a constructor dependency: pass the same collector instance
used for the rest of the pipeline so stage spans nest under the same trace. If
none is provided, a per-instance NullMetricsCollector is used so stages stay
standalone-constructable (useful for tests).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

from moment_to_action.metrics import NullMetricsCollector, SpanType

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class Stage(ABC):
    """Abstract base for all pipeline stages — a lazy generator transformer."""

    def __init__(
        self,
        *,
        window: int = 1,
        stride: int | None = None,
        ready: Callable[[list[Message]], bool] | None = None,
        drop: Callable[[Message], bool] | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Configure windowing, emit/drop predicates, and the metrics collector.

        Args:
            window: Number of buffered messages passed to ``_process`` at once.
                ``1`` (the default) is a plain 1:1 map. Larger values buffer
                that many of the most recent messages (e.g. frame windowing
                for a clip-consuming stage).
            stride: How many new messages must arrive between emissions once
                the buffer is full. Defaults to ``window`` (non-overlapping
                windows). Ignored when ``ready`` is provided.
            ready: Optional predicate over the current buffer contents that
                overrides the count-based emit check (``len(buf) == window``)
                and the ``stride`` gate entirely — use this for custom emit
                conditions (e.g. "scene boundary" or "min_fps elapsed").
            drop: Optional predicate to discard input messages before they
                enter the buffer (e.g. dropped frames, wrong message types).
            metrics: Metrics collector used to time this stage's execution.
                Defaults to a per-instance ``NullMetricsCollector``.

        Raises:
            ValueError: If ``window`` or the effective ``stride`` is less than 1.
        """
        if window < 1:
            msg = f"window must be >= 1, got {window}"
            raise ValueError(msg)
        self._window = window
        self._stride = stride if stride is not None else window
        if self._stride < 1:
            msg = f"stride must be >= 1, got {self._stride}"
            raise ValueError(msg)
        self._ready = ready
        self._drop = drop
        self._metrics = metrics or NullMetricsCollector()

    @property
    def name(self) -> str:
        """Return the class name as the stage identifier."""
        return self.__class__.__name__

    def process(self, stream: Iterator[Message]) -> Generator[Message, None, None]:
        """Lazily consume *stream*, buffering/windowing and delegating to ``_process``.

        Args:
            stream: Iterator of incoming messages (typically the previous
                stage's output, or the sensor stream for the first stage).

        Yields:
            Zero or more output messages per emitted window, produced by
            ``_process``. Each emission is wrapped in its own ``SpanType.STAGE``
            span so nested ``MODEL_*`` spans opened inside ``_process`` attach
            to it.
        """
        buf: deque[Message] = deque(maxlen=self._window)
        new_since_emit = 0
        emitted_once = False
        try:
            for msg in stream:
                if self._drop is not None and self._drop(msg):
                    continue

                buf.append(msg)
                new_since_emit += 1
                items = list(buf)

                if self._ready is not None:
                    if not self._ready(items):
                        continue
                else:
                    if len(items) < self._window:
                        continue
                    if emitted_once and new_since_emit < self._stride:
                        continue

                new_since_emit = 0
                emitted_once = True
                with self._metrics.start_span(SpanType.STAGE, self.name):
                    yield from self._process(items)
        finally:
            # `for msg in stream` (unlike `yield from`) does not forward close()/throw()
            # to *stream* automatically. Closing it explicitly here cascades an early
            # abort (GeneratorExit) up through every upstream stage in the chain.
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    @abstractmethod
    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Transform a buffered window of messages into zero or more outputs.

        Args:
            items: The current window of buffered messages, oldest first.
                Has exactly ``window`` items when using the default count-based
                emit check; may vary in length when a custom ``ready``
                predicate is used.

        Yields:
            Zero or more output messages.
        """
        ...
