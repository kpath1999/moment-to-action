"""Unit tests for prompt_tuning proposers and chat-client ports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from moment_to_action.prompt_tuning import (
    QUIT_TOKEN,
    ChatClient,
    FileBridgeChatClient,
    HumanProposer,
    LLMProposer,
    NotConfiguredChatClient,
    PromptCandidate,
    PromptProposer,
    StopTuning,
    TuningState,
    build_meta_prompt,
    parse_candidate_reply,
)
from moment_to_action.prompt_tuning._proposers import _extract_json_object

from .conftest import make_result, make_scored, make_state

if TYPE_CHECKING:
    from pathlib import Path


def _state_with_failure() -> TuningState:
    """Build a two-candidate state whose best has one failing case."""
    weak = make_scored(PromptCandidate("weak", "t"), [make_result("a", score=0.0, passed=False)])
    strong = make_scored(
        PromptCandidate("strong", "t", generation=1),
        [
            make_result("a", score=1.0, passed=True),
            make_result("b", score=0.0, passed=False, response="wrong", error=""),
        ],
    )
    return make_state(weak, strong, task="detect fights")


@pytest.mark.unit
class TestMetaPrompt:
    """Tests for build_meta_prompt / _format_* helpers."""

    def test_meta_prompt_includes_task_history_and_failures(self) -> None:
        """The meta-prompt embeds task, scorer, prior attempts, and failures."""
        prompt = build_meta_prompt(_state_with_failure())
        assert "detect fights" in prompt
        assert "keyword_recall" in prompt
        assert "system_prompt" in prompt
        # Best attempt's failing case (case 'b', response 'wrong') is shown.
        assert "wrong" in prompt

    def test_meta_prompt_reports_no_failures_when_all_pass(self) -> None:
        """When the best candidate has no failures, the block says so."""
        good = make_scored(PromptCandidate("g", "t"), [make_result("a", score=1.0, passed=True)])
        prompt = build_meta_prompt(make_state(good))
        assert "(no failing cases)" in prompt

    def test_failure_detail_prefers_error_then_response_then_placeholder(self) -> None:
        """Failure rendering shows the error, else response, else a placeholder."""
        errored = make_scored(
            PromptCandidate("e", "t"),
            [make_result("a", score=0.0, passed=False, response="", error="boom")],
        )
        empty = make_scored(
            PromptCandidate("f", "t"),
            [make_result("a", score=0.0, passed=False, response="", error="")],
        )
        assert "boom" in build_meta_prompt(make_state(errored))
        assert "(empty response)" in build_meta_prompt(make_state(empty))


@pytest.mark.unit
class TestJsonExtraction:
    """Tests for _extract_json_object and parse_candidate_reply."""

    def test_extracts_fenced_json(self) -> None:
        """A fenced ```json block is parsed."""
        text = 'prose\n```json\n{"system_prompt": "s"}\n```\ntrailing'
        assert _extract_json_object(text) == {"system_prompt": "s"}

    def test_extracts_braced_json_without_fence(self) -> None:
        """A bare {...} span is parsed when there is no fence."""
        text = 'Here you go: {"task_template": "t {question}"} thanks'
        assert _extract_json_object(text) == {"task_template": "t {question}"}

    def test_raises_when_no_json(self) -> None:
        """Text with no JSON object raises ValueError."""
        with pytest.raises(ValueError, match="could not parse"):
            _extract_json_object("no json here")

    def test_skips_unparseable_brace_span(self) -> None:
        """A brace span that is not valid JSON is skipped, then ValueError raised."""
        with pytest.raises(ValueError, match="could not parse"):
            _extract_json_object("prefix {not: valid, json} suffix")

    def test_parse_candidate_reply_builds_candidate(self) -> None:
        """A valid reply produces a candidate with lineage stamped."""
        reply = '{"system_prompt": "S", "task_template": "T {question}", "rationale": "why"}'
        candidate = parse_candidate_reply(reply, generation=4, parent_id="gen03-abc")
        assert candidate.system_prompt == "S"
        assert candidate.task_template == "T {question}"
        assert candidate.rationale == "why"
        assert candidate.generation == 4
        assert candidate.parent_id == "gen03-abc"

    def test_parse_candidate_reply_defaults_rationale(self) -> None:
        """A missing rationale defaults to empty."""
        candidate = parse_candidate_reply('{"system_prompt": "S"}', generation=1, parent_id="p")
        assert candidate.rationale == ""
        assert candidate.task_template == ""

    def test_parse_candidate_reply_requires_content(self) -> None:
        """A reply with neither prompt field raises ValueError."""
        with pytest.raises(ValueError, match="neither"):
            parse_candidate_reply('{"rationale": "x"}', generation=1, parent_id="p")


@pytest.mark.unit
class TestChatClients:
    """Tests for the ChatClient port implementations."""

    def test_not_configured_client_raises(self) -> None:
        """NotConfiguredChatClient.complete always raises with guidance."""
        client = NotConfiguredChatClient()
        assert isinstance(client, ChatClient)
        with pytest.raises(RuntimeError, match="No LLM ChatClient is configured"):
            client.complete("s", "u")

    def test_file_bridge_writes_request_and_reads_response(self, tmp_path: Path) -> None:
        """FileBridgeChatClient writes the request and reads the reply file back."""
        replies = ["first reply", "second reply"]

        def await_response(request_path: Path, response_path: Path) -> None:
            assert request_path.exists()
            assert "SYSTEM" in request_path.read_text()
            response_path.write_text(replies[int(request_path.stem.split("_")[1]) - 1])

        client = FileBridgeChatClient(bridge_dir=tmp_path, await_response=await_response)
        assert client.complete("sys-a", "user-a") == "first reply"
        assert client.complete("sys-b", "user-b") == "second reply"
        # Counter advanced the filenames.
        assert (tmp_path / "request_01.md").exists()
        assert (tmp_path / "request_02.md").exists()

    def test_file_bridge_raises_when_response_missing(self, tmp_path: Path) -> None:
        """FileBridgeChatClient raises if the reply file is never created."""
        client = FileBridgeChatClient(bridge_dir=tmp_path, await_response=lambda _req, _resp: None)
        with pytest.raises(FileNotFoundError, match="was not created"):
            client.complete("s", "u")


class _StubChatClient:
    """A ChatClient that returns a fixed reply and records the messages."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.system = ""
        self.user = ""

    def complete(self, system: str, user: str) -> str:
        """Record the messages and return the canned reply."""
        self.system = system
        self.user = user
        return self.reply


@pytest.mark.unit
class TestLLMProposer:
    """Tests for LLMProposer."""

    def test_propose_parses_reply_and_stamps_lineage(self) -> None:
        """Propose sends the meta-prompt and parses the reply into a candidate."""
        reply = '{"system_prompt": "NEW", "task_template": "T {question}", "rationale": "r"}'
        client = _StubChatClient(reply)
        proposer = LLMProposer(client=client)
        assert isinstance(proposer, PromptProposer)
        state = _state_with_failure()

        candidate = proposer.propose(state)

        assert candidate.system_prompt == "NEW"
        # Generation is latest + 1; parent is the best candidate's label.
        assert candidate.generation == state.latest.candidate.generation + 1
        assert candidate.parent_id == state.best.candidate.label
        assert "detect fights" in client.user


@pytest.mark.unit
class TestHumanProposer:
    """Tests for HumanProposer."""

    def _proposer(self, answers: list[str]) -> tuple[HumanProposer, list[str]]:
        """Build a HumanProposer whose read() replays ``answers`` and records writes."""
        written: list[str] = []
        answers_iter = iter(answers)
        return (
            HumanProposer(read=lambda _label: next(answers_iter), write=written.append),
            written,
        )

    def test_blank_answers_reuse_best_and_stop(self) -> None:
        """Blank system + template reuse the best and raise StopTuning (no change)."""
        proposer, written = self._proposer(["", "", ""])
        state = _state_with_failure()
        with pytest.raises(StopTuning, match="no changes"):
            proposer.propose(state)
        # The best prompt and its failing cases were shown to the user.
        assert any("Best so far" in line for line in written)

    def test_quit_token_stops(self) -> None:
        """Entering the quit token for the system prompt raises StopTuning."""
        proposer, _ = self._proposer([QUIT_TOKEN])
        with pytest.raises(StopTuning):
            proposer.propose(_state_with_failure())

    def test_edits_produce_new_candidate(self) -> None:
        """Non-blank answers build a new candidate with lineage and rationale."""
        proposer, _ = self._proposer(["NEW SYS", "NEW TMPL {question}", "because"])
        state = _state_with_failure()
        candidate = proposer.propose(state)
        assert candidate.system_prompt == "NEW SYS"
        assert candidate.task_template == "NEW TMPL {question}"
        assert candidate.rationale == "because"
        assert candidate.parent_id == state.best.candidate.label

    def test_blank_template_reuses_best_template(self) -> None:
        """A blank template answer keeps the best candidate's template."""
        proposer, _ = self._proposer(["NEW SYS", "", ""])
        state = _state_with_failure()
        candidate = proposer.propose(state)
        assert candidate.system_prompt == "NEW SYS"
        assert candidate.task_template == state.best.candidate.task_template
