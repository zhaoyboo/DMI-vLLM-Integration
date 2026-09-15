"""Negative controls for the bounded GPU workload's result comparator."""

from copy import deepcopy

import pytest
import torch

from tests.v029_smoke import compare


@pytest.mark.parametrize("fault", [
    None, "public", "runner", "configuration", "versions", "logits",
    "missing_step", "missing_storage", "missing_configuration",
])
def test_smoke_comparator_rejects_false_green(tmp_path, fault):
    stock = {
        "public": [{"request_id": "A", "token_ids": [1, 2], "finish_reason": "length"}],
        "runner": ["vllm.v1.worker.gpu.model_runner"],
        "configuration": {"custom_ops": ["none"], "max_tokens": 8},
        "versions": {"vllm": "0.29.0"},
        "logits": [torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)],
        "storage_rows": 0,
    }
    monitored = deepcopy(stock)
    monitored["storage_rows"] = 1
    if fault in {"public", "runner", "configuration", "versions"}:
        monitored[fault] = "wrong"
    elif fault == "logits":
        # Same argmax and public tokens, but not the same logits: must fail.
        monitored["logits"][0][0, 0] = 1.5
    elif fault == "missing_step":
        monitored["logits"] = []
    elif fault == "missing_storage":
        monitored["storage_rows"] = 0
    elif fault == "missing_configuration":
        del monitored["configuration"]
    paths = [tmp_path / "stock.pt", tmp_path / "monitored.pt"]
    for value, path in zip((stock, monitored), paths):
        torch.save(value, path)
    if fault is None:
        compare(*paths)
    else:
        with pytest.raises((AssertionError, KeyError)):
            compare(*paths)
