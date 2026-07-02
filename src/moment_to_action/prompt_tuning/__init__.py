"""Prompt-tuning infrastructure for the VLM pipeline.

This subpackage implements the closed iteration loop *prompt → run pipeline →
result → iterate prompt*, with the "iterate" step pluggable between a human and
an online LLM (see :mod:`moment_to_action.prompt_tuning._proposers`).

Typical wiring (see ``scripts/tune_vlm_prompt.py`` for a runnable driver)::

    runner = PromptRunner(target=VLMResponseTarget(model), scorer=KeywordRecallScorer())
    tuner = PromptTuner(
        runner=runner,
        proposer=HumanProposer(read=..., write=...),
        dataset=dataset,
        task_description="...",
        store=TrajectoryStore(run_dir),
    )
    final_state = tuner.run(seed, max_iterations=10, target_score=0.9)
    best = final_state.best

The public surface is intentionally small; concrete VLM wiring and the
interactive loop live in the driver script, while everything importable here is
pure and unit-tested.
"""

from __future__ import annotations

from ._proposers import (
    OPTIMIZER_SYSTEM_PROMPT,
    QUIT_TOKEN,
    ChatClient,
    FileBridgeChatClient,
    HumanProposer,
    LLMProposer,
    NotConfiguredChatClient,
    PromptProposer,
    StopTuning,
    build_meta_prompt,
    parse_candidate_reply,
)
from ._runner import PromptRunner
from ._scoring import KeywordRecallScorer, LabelMatchScorer, Scorer
from ._store import (
    TrajectoryStore,
    candidate_to_dict,
    render_prompt,
    report_to_dict,
    scored_to_dict,
)
from ._target import MultimodalModel, ResponseTarget, VLMResponseTarget
from ._tuner import PromptTuner
from ._types import (
    QUESTION_PLACEHOLDER,
    CaseResult,
    EvalCase,
    EvalDataset,
    EvalReport,
    PromptCandidate,
    ScoredCandidate,
    TuningState,
)

__all__ = [
    "OPTIMIZER_SYSTEM_PROMPT",
    "QUESTION_PLACEHOLDER",
    "QUIT_TOKEN",
    "CaseResult",
    "ChatClient",
    "EvalCase",
    "EvalDataset",
    "EvalReport",
    "FileBridgeChatClient",
    "HumanProposer",
    "KeywordRecallScorer",
    "LLMProposer",
    "LabelMatchScorer",
    "MultimodalModel",
    "NotConfiguredChatClient",
    "PromptCandidate",
    "PromptProposer",
    "PromptRunner",
    "PromptTuner",
    "ResponseTarget",
    "ScoredCandidate",
    "Scorer",
    "StopTuning",
    "TrajectoryStore",
    "TuningState",
    "VLMResponseTarget",
    "build_meta_prompt",
    "candidate_to_dict",
    "parse_candidate_reply",
    "render_prompt",
    "report_to_dict",
    "scored_to_dict",
]
