"""Proposers — the "iterate the prompt" step, for humans and online LLMs alike.

A proposer reads the :class:`~moment_to_action.prompt_tuning._types.TuningState`
(the scored trajectory so far) and returns the next
:class:`~moment_to_action.prompt_tuning._types.PromptCandidate`.  This is the
extension **port**: the tuner depends only on the :class:`PromptProposer`
protocol, so a human editor, a local heuristic, or a remote LLM are
interchangeable.

Two proposers ship today:

* :class:`HumanProposer` — collects a new prompt via injected read/write
  callbacks (the driver wires these to the terminal / ``$EDITOR``).
* :class:`LLMProposer` — builds an OPRO-style meta-prompt from the trajectory,
  sends it through a :class:`ChatClient`, and parses the reply.

The :class:`ChatClient` port has three implementations:

* :class:`NotConfiguredChatClient` — a placeholder that errors clearly; swap in
  a real HTTP client by implementing :meth:`ChatClient.complete`.
* :class:`FileBridgeChatClient` — writes the meta-prompt to a file and reads the
  reply back from a sibling file, so an LLM can be kept "in the loop" by hand
  with no API today.

Raising :class:`StopTuning` from any proposer ends the loop gracefully.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import attrs

from ._types import PromptCandidate

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ._types import EvalReport, TuningState

QUIT_TOKEN = "/stop"  # noqa: S105 — a UI sentinel, not a secret
"""Sentinel a human can enter for the system prompt to end the tuning loop."""

OPTIMIZER_SYSTEM_PROMPT = (
    "You are a prompt-optimization engine. You improve the system prompt and "
    "task template used to drive a small on-device vision-language model on a "
    "fixed classification task. You are given the task description, the scoring "
    "rubric, previously tried prompts with their scores (worst first, best "
    "last), and the failing cases of the best prompt so far. Propose ONE new "
    "prompt that should score higher. Change the wording deliberately based on "
    "the observed failures; do not merely rephrase the best prompt. Respond "
    "with a single JSON object and nothing else, using exactly these keys: "
    '"system_prompt" (string), "task_template" (string, may contain the token '
    '{question} where the per-case question is inserted), and "rationale" '
    "(string, one sentence on what you changed and why)."
)
"""System prompt used to drive an LLM proposer."""


class StopTuning(Exception):  # noqa: N818 — a control-flow signal, not an error
    """Raised by a proposer to end the tuning loop gracefully."""


@runtime_checkable
class PromptProposer(Protocol):
    """Proposes the next candidate given the trajectory so far."""

    def propose(self, state: TuningState) -> PromptCandidate:
        """Return the next candidate to evaluate.

        Args:
            state: The scored trajectory so far.

        Returns:
            The next :class:`PromptCandidate` to evaluate.

        Raises:
            StopTuning: To end the loop without proposing.
        """
        ...


@runtime_checkable
class ChatClient(Protocol):
    """A minimal single-turn chat completion port for LLM proposers."""

    def complete(self, system: str, user: str) -> str:
        """Return the model's reply to a system + user message.

        Args:
            system: The system message.
            user: The user message.

        Returns:
            The model's raw text reply.
        """
        ...


# ---------------------------------------------------------------------------
# Trajectory rendering helpers (shared by the meta-prompt and human summary)
# ---------------------------------------------------------------------------


def _format_failures(report: EvalReport, limit: int) -> str:
    """Render the worst failing cases of a report as a readable block.

    Args:
        report: The report whose failures to render.
        limit: Maximum number of failures to include.

    Returns:
        A formatted multi-line string, or a note when there are no failures.
    """
    failures = report.failures[:limit]
    if not failures:
        return "(no failing cases)"
    lines: list[str] = []
    for i, r in enumerate(failures, start=1):
        detail = r.error or r.response.strip() or "(empty response)"
        lines.append(
            f"{i}. [{r.app or 'general'}] Q: {r.question}\n"
            f"   expected: {r.expected_label or '(any)'}  score: {r.score:.2f}\n"
            f"   model said: {detail}"
        )
    return "\n".join(lines)


def _format_history(state: TuningState, limit: int) -> str:
    """Render the top scored candidates, worst first (OPRO ordering).

    Args:
        state: The trajectory to summarize.
        limit: Maximum number of candidates to include.

    Returns:
        A formatted multi-line string of prior attempts and their scores.
    """
    top = state.top_k(limit)
    # Best last so the LLM anchors on the strongest example most recently seen.
    ordered = sorted(top, key=lambda sc: sc.report.mean_score)
    lines: list[str] = []
    for sc in ordered:
        rep = sc.report
        lines.append(
            f"--- attempt {sc.candidate.label} "
            f"(mean_score={rep.mean_score:.3f}, pass_rate={rep.pass_rate:.2f}) ---\n"
            f"system_prompt: {sc.candidate.system_prompt!r}\n"
            f"task_template: {sc.candidate.task_template!r}"
        )
    return "\n\n".join(lines)


def build_meta_prompt(state: TuningState, *, max_history: int = 6, max_failures: int = 6) -> str:
    """Build the OPRO-style user message asking an LLM for a better prompt.

    Args:
        state: The scored trajectory so far.
        max_history: Maximum number of prior attempts to include.
        max_failures: Maximum number of failing cases (of the best attempt) to show.

    Returns:
        A user-message string ready to hand to a :class:`ChatClient`.
    """
    best = state.best
    return (
        f"# Task\n{state.task_description}\n\n"
        f"# Scoring\nEach case is scored in [0, 1] by the "
        f"'{best.report.scorer_name}' scorer; a case passes at "
        f">= {best.report.pass_threshold:.2f}. Maximize the mean score.\n\n"
        f"# Previously tried prompts (worst first, best last)\n"
        f"{_format_history(state, max_history)}\n\n"
        f"# Failing cases of the best prompt so far "
        f"({best.candidate.label}, mean_score={best.report.mean_score:.3f})\n"
        f"{_format_failures(best.report, max_failures)}\n\n"
        f"# Your task\nPropose ONE improved prompt as a single JSON object with "
        f'keys "system_prompt", "task_template", and "rationale".'
    )


# ---------------------------------------------------------------------------
# Chat clients (the online-LLM port)
# ---------------------------------------------------------------------------


@attrs.frozen
class NotConfiguredChatClient:
    """Placeholder :class:`ChatClient` used when no LLM API is wired up yet."""

    def complete(self, system: str, user: str) -> str:
        """Always raise — no backend is configured.

        Args:
            system: Ignored.
            user: Ignored.

        Raises:
            RuntimeError: Always, with guidance on how to configure a client.
        """
        del system, user
        msg = (
            "No LLM ChatClient is configured. Implement ChatClient.complete "
            "(e.g. an HTTP call to your model endpoint) and pass it to "
            "LLMProposer, or use FileBridgeChatClient for a manual/offline loop."
        )
        raise RuntimeError(msg)


@attrs.define
class FileBridgeChatClient:
    """Bridges to any LLM by writing the request and reading the reply from files.

    Each call writes ``request_NN.md`` into ``bridge_dir`` and expects the reply
    at ``response_NN.txt``.  The ``await_response`` callback is responsible for
    blocking until the reply file exists (the driver prints instructions and
    waits for the user; tests inject the file directly).

    Attributes:
        bridge_dir: Directory used to exchange request/response files.
        await_response: Callback ``(request_path, response_path) -> None`` that
            blocks until ``response_path`` has been written.
    """

    bridge_dir: Path
    await_response: Callable[[Path, Path], None]
    _counter: int = attrs.field(default=0, init=False)

    def complete(self, system: str, user: str) -> str:
        """Write the request, wait for the reply file, and return its contents.

        Args:
            system: The system message.
            user: The user message.

        Returns:
            The text read back from the response file.

        Raises:
            FileNotFoundError: If the response file is still absent after
                ``await_response`` returns.
        """
        self._counter += 1
        n = self._counter
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        request_path = self.bridge_dir / f"request_{n:02d}.md"
        response_path = self.bridge_dir / f"response_{n:02d}.txt"
        request_path.write_text(f"# SYSTEM\n\n{system}\n\n# USER\n\n{user}\n")
        self.await_response(request_path, response_path)
        if not response_path.exists():
            msg = f"expected LLM reply at {response_path}, but it was not created"
            raise FileNotFoundError(msg)
        return response_path.read_text()


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict[str, object]:
    """Extract the first JSON object from an LLM reply.

    Tries a fenced ```json block first, then the outermost ``{...}`` span.

    Args:
        text: The raw LLM reply.

    Returns:
        The decoded JSON object.

    Raises:
        ValueError: If no JSON object can be parsed from ``text``.
    """
    snippets: list[str] = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        snippets.append(fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        snippets.append(text[start : end + 1])
    for snippet in snippets:
        try:
            obj = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    msg = "could not parse a JSON object from the LLM reply"
    raise ValueError(msg)


def parse_candidate_reply(reply: str, *, generation: int, parent_id: str) -> PromptCandidate:
    """Parse an LLM reply into a :class:`PromptCandidate`.

    Args:
        reply: The raw LLM reply, expected to contain a JSON object with
            ``system_prompt``, ``task_template`` and (optionally) ``rationale``.
        generation: Generation number to stamp on the new candidate.
        parent_id: :attr:`PromptCandidate.label` of the parent candidate.

    Returns:
        The parsed candidate.

    Raises:
        ValueError: If no JSON object is found, or it defines neither a system
            prompt nor a task template.
    """
    obj = _extract_json_object(reply)
    system_prompt = str(obj.get("system_prompt", ""))
    task_template = str(obj.get("task_template", ""))
    rationale = str(obj.get("rationale", ""))
    if not system_prompt and not task_template:
        msg = "LLM reply defined neither 'system_prompt' nor 'task_template'"
        raise ValueError(msg)
    return PromptCandidate(
        system_prompt=system_prompt,
        task_template=task_template,
        generation=generation,
        parent_id=parent_id,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Proposers
# ---------------------------------------------------------------------------


@attrs.define
class LLMProposer:
    """Proposes candidates by prompting an LLM through a :class:`ChatClient`.

    Attributes:
        client: The chat client used to obtain proposals.
        system_prompt: System message driving the optimizer LLM.
        max_history: Maximum prior attempts included in the meta-prompt.
        max_failures: Maximum failing cases included in the meta-prompt.
    """

    client: ChatClient
    system_prompt: str = OPTIMIZER_SYSTEM_PROMPT
    max_history: int = 6
    max_failures: int = 6

    def propose(self, state: TuningState) -> PromptCandidate:
        """Ask the LLM for an improved candidate.

        Args:
            state: The scored trajectory so far.

        Returns:
            The parsed :class:`PromptCandidate`.

        Raises:
            ValueError: If the LLM reply cannot be parsed into a candidate.
        """
        user = build_meta_prompt(
            state, max_history=self.max_history, max_failures=self.max_failures
        )
        reply = self.client.complete(self.system_prompt, user)
        return parse_candidate_reply(
            reply,
            generation=state.latest.candidate.generation + 1,
            parent_id=state.best.candidate.label,
        )


@attrs.define
class HumanProposer:
    """Proposes candidates by collecting edits from a person.

    The ``read`` callback is asked for a new system prompt, task template, and
    rationale in turn; a blank answer reuses the corresponding field of the
    current best candidate, and entering :data:`QUIT_TOKEN` for the system
    prompt ends the loop.  ``write`` is used to show the current best prompt and
    its failing cases for context.

    Attributes:
        read: Callback ``(field_label) -> user_input`` (e.g. ``input``).
        write: Callback ``(message) -> None`` (e.g. ``print``).
        max_failures: Maximum failing cases shown for context.
    """

    read: Callable[[str], str]
    write: Callable[[str], None]
    max_failures: int = 6

    def propose(self, state: TuningState) -> PromptCandidate:
        """Collect a new candidate from the user.

        Args:
            state: The scored trajectory so far.

        Returns:
            The edited :class:`PromptCandidate`.

        Raises:
            StopTuning: If the user enters :data:`QUIT_TOKEN` or makes no change.
        """
        base = state.best.candidate
        self.write(
            f"Best so far: {base.label} "
            f"(mean_score={state.best.report.mean_score:.3f}, "
            f"pass_rate={state.best.report.pass_rate:.2f})\n"
            f"system_prompt: {base.system_prompt!r}\n"
            f"task_template: {base.task_template!r}\n"
            f"Failing cases:\n{_format_failures(state.best.report, self.max_failures)}\n"
            f"(blank reuses the best value; enter {QUIT_TOKEN!r} to finish)"
        )
        raw_system = self.read("system prompt")
        if raw_system.strip() == QUIT_TOKEN:
            raise StopTuning
        raw_template = self.read("task template")
        rationale = self.read("rationale")
        system_prompt = base.system_prompt if raw_system == "" else raw_system
        task_template = base.task_template if raw_template == "" else raw_template
        if system_prompt == base.system_prompt and task_template == base.task_template:
            msg = "no changes made to the best prompt"
            raise StopTuning(msg)
        return PromptCandidate(
            system_prompt=system_prompt,
            task_template=task_template,
            generation=state.latest.candidate.generation + 1,
            parent_id=base.label,
            rationale=rationale,
        )
