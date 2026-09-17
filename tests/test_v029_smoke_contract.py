"""Negative controls for the bounded GPU workload's result comparator."""

from copy import deepcopy

import pytest
import torch

from tests.v029_smoke import compare, require_multiprocess_engine
from tests.v029_residual_reference import compare_residual_rows, old_residual_expression


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


@pytest.mark.parametrize("fault", [None, "value", "missing", "duplicate", "dtype", "shape", "nonfinite", "identity"])
def test_residual_oracle_rejects_false_green(fault):
    expected = [{"request_id": "A", "act_name": "blocks.hook_resid_mid",
                 "layer_no": 0, "start": 0, "end": 1,
                 "tensor": torch.tensor([[1., 2.]], dtype=torch.bfloat16)}]
    observed = deepcopy(expected)
    if fault == "value":
        observed[0]["tensor"][0, 0] = 1.5
    elif fault == "missing":
        observed = []
    elif fault == "duplicate":
        observed += deepcopy(observed)
    elif fault == "dtype":
        observed[0]["tensor"] = observed[0]["tensor"].float()
    elif fault == "shape":
        observed[0]["tensor"] = observed[0]["tensor"].flatten()
    elif fault == "nonfinite":
        observed[0]["tensor"][0, 0] = float("nan")
    elif fault == "identity":
        observed[0]["request_id"] = "B"
    if fault is None:
        assert compare_residual_rows(expected, observed) == 1
    else:
        with pytest.raises(AssertionError):
            compare_residual_rows(expected, observed)


@pytest.mark.parametrize("has_residual", [False, True])
def test_old_expression_reference_owns_pre_update_snapshot(has_residual):
    hidden = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
    residual = torch.tensor([[3., 4.]], dtype=torch.bfloat16)
    snapshot = old_residual_expression((hidden, residual) if has_residual else (hidden,))
    hidden.zero_()
    residual.zero_()
    assert snapshot.tolist() == ([[4., 6.]] if has_residual else [[1., 2.]])


def test_smoke_barrier_rejects_inprocess_engine(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with pytest.raises(RuntimeError, match="EngineCoreProc-only"):
        require_multiprocess_engine()


@pytest.mark.parametrize("clear_path", [False, True])
def test_release_wrapper_library_path_is_opt_in(tmp_path, clear_path):
    """Exercise the shell wrapper without importing vLLM or running any GPU."""
    import os
    from pathlib import Path
    import subprocess

    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        '#!/bin/sh\n'
        'printf "%s\\n" "${LD_LIBRARY_PATH-unset}" >> "$TEST_ENV_LOG"\n'
        'echo "Directly load AOT compilation"\n'
    )
    fake_python.chmod(0o700)
    log = tmp_path / "environment.log"
    env = dict(os.environ, LD_LIBRARY_PATH="/example/required-cuda-library",
               DMI_V029_CLEAR_LD_LIBRARY_PATH="1" if clear_path else "0",
               DMI_V029_PYTHON=str(fake_python), DMI_V029_ARTIFACT_ROOT=str(tmp_path),
               TEST_ENV_LOG=str(log))
    subprocess.run(["bash", "tests/run_v029_smoke.sh"],
                   cwd=Path(__file__).parents[1], env=env,
                   capture_output=True, text=True, timeout=20, check=True)
    observed = log.read_text().splitlines()
    assert len(observed) == 18  # four main cells, AOT reload, residual; 3 processes each
    assert set(observed) == {"unset" if clear_path else "/example/required-cuda-library"}
