"""Unit tests for prompt_tuning scorers."""

from __future__ import annotations

import pytest

from moment_to_action.prompt_tuning import KeywordRecallScorer, LabelMatchScorer

from .conftest import make_case


@pytest.mark.unit
class TestKeywordRecallScorer:
    """Tests for KeywordRecallScorer."""

    def test_name(self) -> None:
        """The scorer identifies itself as 'keyword_recall'."""
        assert KeywordRecallScorer().name == "keyword_recall"

    def test_no_keywords_scores_one(self) -> None:
        """A case with no keywords always scores 1.0."""
        case = make_case(keywords=())
        assert KeywordRecallScorer().score("anything", case) == 1.0

    def test_partial_recall(self) -> None:
        """Recall is the fraction of keywords present, case-insensitively."""
        case = make_case(keywords=("fight", "violence"))
        assert KeywordRecallScorer().score("a FIGHT broke out", case) == 0.5

    def test_full_recall(self) -> None:
        """All keywords present yields 1.0."""
        case = make_case(keywords=("fight", "violence"))
        assert KeywordRecallScorer().score("fight and violence", case) == 1.0


@pytest.mark.unit
class TestLabelMatchScorer:
    """Tests for LabelMatchScorer."""

    def test_name(self) -> None:
        """The scorer identifies itself as 'label_match'."""
        assert LabelMatchScorer().name == "label_match"

    def test_empty_label_scores_one(self) -> None:
        """A case with no expected label always scores 1.0."""
        case = make_case(expected="")
        assert LabelMatchScorer().score("whatever", case) == 1.0

    def test_whole_word_match_case_insensitive(self) -> None:
        """The label matches as a whole word, ignoring case."""
        case = make_case(expected="YES")
        assert LabelMatchScorer().score("The answer is yes.", case) == 1.0

    def test_no_substring_match(self) -> None:
        """The label does not match inside a larger word."""
        case = make_case(expected="NO")
        assert LabelMatchScorer().score("there is a lot of NOISE here", case) == 0.0

    def test_label_with_regex_metacharacters(self) -> None:
        """A label containing regex metacharacters is matched literally."""
        case = make_case(expected="non-compliant")
        assert LabelMatchScorer().score("Result: NON-COMPLIANT overall", case) == 1.0
