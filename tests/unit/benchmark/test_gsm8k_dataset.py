from __future__ import annotations

import pytest

from moment_to_action.benchmark._gsm8k_dataset import GSM8KDataset


@pytest.mark.unit
def test_gsm8k_dataset_extracts_numeric_answers() -> None:
    assert GSM8KDataset.extract_numeric_answer("work\n#### 42") == "42"
    assert GSM8KDataset.extract_numeric_answer("value: 1,234") == "1234"
    assert GSM8KDataset.extract_numeric_answer("final 7.0") == "7"


@pytest.mark.unit
def test_gsm8k_dataset_rejects_non_positive_n_items() -> None:
    with pytest.raises(ValueError, match="n_items must be greater than 0"):
        GSM8KDataset(n_items=0)
