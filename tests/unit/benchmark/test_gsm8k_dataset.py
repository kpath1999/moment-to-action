from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from moment_to_action.benchmark._datasets._gsm8k_dataset import GSM8KDataset


@pytest.mark.unit
def test_gsm8k_dataset_extracts_numeric_answers() -> None:
    assert GSM8KDataset.extract_numeric_answer("work\n#### 42") == "42"
    assert GSM8KDataset.extract_numeric_answer("value: 1,234") == "1234"
    assert GSM8KDataset.extract_numeric_answer("final 7.0") == "7"
    assert GSM8KDataset.extract_numeric_answer("prefix 8 #### no-number") == "8"


@pytest.mark.unit
def test_gsm8k_dataset_rejects_non_positive_n_items() -> None:
    with pytest.raises(ValueError, match="n_items must be greater than 0"):
        GSM8KDataset(n_items=0)


@pytest.mark.unit
def test_gsm8k_parse_row_filters_invalid_inputs() -> None:
    assert GSM8KDataset._parse_row("not-dict") is None
    assert GSM8KDataset._parse_row({"question": 1, "answer": "#### 2"}) is None
    assert GSM8KDataset._parse_row({"question": "Q", "answer": 2}) is None
    assert GSM8KDataset._parse_row({"question": "Q", "answer": "no-number"}) is None


@pytest.mark.unit
def test_gsm8k_load_items_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_rows = [
        {"question": "Q1", "answer": "#### 1"},
        {"question": "Q2", "answer": "2"},
        {"question": "bad", "answer": "none"},
    ]

    fake_datasets = mock.MagicMock()
    fake_datasets.load_dataset.return_value = fake_rows
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    dataset = GSM8KDataset(n_items=2)
    assert len(dataset.items()) == 2
    assert dataset.dataset_name == "gsm8k_test"


@pytest.mark.unit
def test_gsm8k_load_items_raises_when_no_valid_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_datasets = mock.MagicMock()
    fake_datasets.load_dataset.return_value = [{"question": "Q", "answer": "n/a"}]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="No valid GSM8K items"):
        GSM8KDataset(n_items=1)


@pytest.mark.unit
def test_gsm8k_load_items_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace())

    with pytest.raises(RuntimeError, match="datasets package is required"):
        GSM8KDataset(n_items=1)


@pytest.mark.unit
def test_gsm8k_normalize_number_non_numeric_and_float() -> None:
    assert GSM8KDataset.normalize_number("abc") == "abc"
    assert GSM8KDataset.normalize_number("3.25") == "3.25"
