"""GBNF grammars for constraining LLM generation via llama.cpp's ``grammar`` field."""

from __future__ import annotations

YES_NO_GRAMMAR = 'root ::= decision rest\ndecision ::= "YES" | "NO"\nrest ::= [^\\x00]*\n'
"""Forces generation to start with a literal ``YES`` or ``NO`` token, followed by
unconstrained free text (the rationale). Pass to ``LlamaGGUFModel.stream(...,
grammar=YES_NO_GRAMMAR)`` so :class:`~moment_to_action.stages.llm.DecisionStage` can
read the leading token unambiguously as soon as it arrives.
"""
