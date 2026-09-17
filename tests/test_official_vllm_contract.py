"""Portable checks against the installed official vLLM release."""

from __future__ import annotations

import inspect
from importlib import import_module
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

import pytest
import vllm
from vllm.v1.worker.gpu_worker import Worker


def _native_extension_spec():
    for module_name in ("vllm._C", "vllm._C_stable_libtorch"):
        spec = find_spec(module_name)
        if spec is not None and spec.origin is not None:
            return spec
    raise AssertionError("official vLLM native extension is not installed")


def test_official_vllm_release_and_native_extension_are_coinstalled() -> None:
    native_spec = _native_extension_spec()

    assert version("vllm") == "0.29.0"
    assert Path(vllm.__file__).resolve().parent == Path(
        native_spec.origin
    ).resolve().parent
    assert "dmi_vllm_integration" not in Path(vllm.__file__).parts


@pytest.mark.gpu
def test_official_vllm_native_extension_loads_on_gpu_runner() -> None:
    native_spec = _native_extension_spec()
    native_extension = import_module(native_spec.name)

    assert Path(native_extension.__file__).resolve() == Path(
        native_spec.origin
    ).resolve()


def test_worker_lifecycle_signatures_match_the_integration_contract() -> None:
    load_model = inspect.signature(Worker.load_model)
    assert load_model.parameters["load_dummy_weights"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert hasattr(Worker, "init_device")
    assert hasattr(Worker, "compile_or_warm_up_model")
    assert hasattr(Worker, "execute_model")
    assert hasattr(Worker, "shutdown")


def test_v029_sampling_guards_match_real_upstream_attributes() -> None:
    """C06/S03: real objects, not fakes that perpetuate a renamed attribute."""
    from vllm.config import ParallelConfig
    from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
    from vllm import SamplingParams

    assert hasattr(ParallelConfig(), "enable_batch_sharded_sampling")
    worker = PromptLogprobsWorker(max_num_reqs=2)  # CPU NumPy state only.
    assert worker.in_progress_prompt_logprobs == {}
    worker.add_request("probe", 0, SamplingParams(prompt_logprobs=1))
    assert "probe" in worker.in_progress_prompt_logprobs
    worker.remove_request("probe")
    assert worker.in_progress_prompt_logprobs == {}


def test_removed_olmo3_is_not_a_production_or_oracle_target() -> None:
    from dmi_vllm_integration.architectures import ARCHITECTURE_REMAP
    from tests.oracles import _ORACLE_MODELS

    assert "Olmo3ForCausalLM" not in ARCHITECTURE_REMAP
    assert not any("olmo3" in value.lower() for value in _ORACLE_MODELS.values())
    root = Path(__file__).parents[1]
    assert not (root / "src/dmi_vllm_integration/models/olmo3.py").exists()
    assert not (root / "tests/oracles/olmo3_compare.py").exists()
