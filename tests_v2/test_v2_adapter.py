"""Portable contracts for the isolated vLLM V2 implementation."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from dmi_vllm_integration.v2.adapter import (
    DMXV2GPUWorker,
    VLLMAdaptor,
    VLLMStepPhase,
    VLLMValidationMode,
    _VLLMRoleFormula,
    _VLLMStepState,
)


def _scheduler() -> SimpleNamespace:
    return SimpleNamespace(
        num_scheduled_tokens={"A": 2, "B": 3},
        total_num_scheduled_tokens=5,
    )


def _worker_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="generate",
            model_impl="auto",
            enable_prompt_embeds=False,
            is_multimodal_model=False,
            is_encoder_decoder=False,
            requires_raw_input_tokens=False,
            architecture="LlamaForCausalLM",
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
            enable_elastic_ep=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=None,
        additional_config={"dmx_hook_selection": "resid_pre"},
    )


def test_v2_worker_rejects_v1_and_speculative_decoding() -> None:
    worker = DMXV2GPUWorker.__new__(DMXV2GPUWorker)
    worker.vllm_config = _worker_config()
    worker.use_v2_model_runner = False
    with pytest.raises(RuntimeError, match="requires.*V2"):
        worker._validate_dmi_config()

    worker.use_v2_model_runner = True
    worker._validate_dmi_config()
    worker.vllm_config.speculative_config = object()
    with pytest.raises(RuntimeError, match="V2.*speculative"):
        worker._validate_dmi_config()


def test_v2_records_prepared_layout_and_input_dtype() -> None:
    scheduler_output = _scheduler()
    adaptor = VLLMAdaptor.__new__(VLLMAdaptor)
    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.ARMED,
        scheduler_output=scheduler_output,
    )
    input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["B", "A"],
        num_scheduled_tokens=np.array([3, 2], dtype=np.int32),
        num_computed_tokens_np=np.array([7, 11], dtype=np.int32),
        num_tokens=5,
    )

    adaptor._record_v2_real_layout(scheduler_output, input_batch)

    layout = adaptor._step_state.layout
    assert layout is not None
    assert layout.raw_req_ids == ("B", "A")
    assert layout.scheduled_counts == (3, 2)
    assert layout.computed_counts == (7, 11)
    assert layout.token_ranges == ((7, 10), (11, 13))
    assert layout.dim0_offsets == (0, 3)
    assert layout.total_rows == 5
    assert VLLMAdaptor._input_ids_dtype(
        SimpleNamespace(
            input_buffers=SimpleNamespace(
                input_ids=torch.empty(0, dtype=torch.int32),
            )
        )
    ) is torch.int32


@pytest.mark.parametrize("selection", ["final_logits", "vllm-full"])
def test_v2_sharded_sampling_rejects_final_logits_before_device_init(selection):
    worker = DMXV2GPUWorker.__new__(DMXV2GPUWorker)
    worker.vllm_config = _worker_config()
    worker.use_v2_model_runner = True
    worker.vllm_config.parallel_config.enable_batch_sharded_sampling = True
    worker.vllm_config.additional_config["dmx_hook_selection"] = selection
    with pytest.raises(RuntimeError, match="batch-sharded"):
        worker._validate_dmi_config()
    # This setting only changes sampling rows, not the decoder's input rows.
    worker.vllm_config.additional_config["dmx_hook_selection"] = "resid_pre"
    worker._validate_dmi_config()
    assert worker.vllm_config.parallel_config.enable_batch_sharded_sampling is True


def test_v2_existing_prompt_logprobs_reject_final_logits(monkeypatch):
    from vllm.v1.worker.gpu_worker import Worker

    worker = DMXV2GPUWorker.__new__(DMXV2GPUWorker)
    worker.adaptor = SimpleNamespace(
        engine=SimpleNamespace(capture_enabled=True),
        _has_global_hooks=True,
        _captures_final_logits=True,
        _step_state=_VLLMStepState(),
    )
    worker.model_runner = SimpleNamespace(prompt_logprobs_worker=SimpleNamespace(
        in_progress_prompt_logprobs={"existing": []},
    ))
    scheduler = _scheduler()
    scheduler.scheduled_new_reqs = []
    monkeypatch.setattr("dmi_vllm_integration.v2.adapter.has_ec_transfer", lambda: False)
    called = []
    monkeypatch.setattr(Worker, "execute_model", lambda *_args: called.append(True))
    with pytest.raises(RuntimeError, match="prompt_logprobs"):
        worker.execute_model(scheduler)
    assert not called


def test_v2_preflight_uses_real_dispatch_descriptor() -> None:
    candidate = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=4,
        uniform_token_count=1,
    )
    adaptor = VLLMAdaptor.__new__(VLLMAdaptor)
    adaptor._step_state = _VLLMStepState(phase=VLLMStepPhase.ARMED)
    adaptor._byte_capacity = 128
    adaptor._role_formulas = (
        _VLLMRoleFormula(execution_terms=((16, 1),), hook_count=1),
    )

    assert not adaptor._preflight_v2_force_eager(
        candidate,
        num_tokens=5,
        num_reqs=2,
    )
    assert adaptor._step_state.capacity_candidate == (
        CUDAGraphMode.FULL,
        candidate,
    )
    adaptor._byte_capacity = 127
    assert adaptor._preflight_v2_force_eager(
        candidate,
        num_tokens=5,
        num_reqs=2,
    )


class _FakeAdaptor:
    def __init__(self, events: list[tuple], force_eager: bool = False) -> None:
        self.events = events
        self.force_eager = force_eager
        self._step_state = _VLLMStepState()
        self._validation_mode = VLLMValidationMode.OFF

    def _record_v2_real_layout(self, scheduler_output, input_batch) -> None:
        self.events.append(("record", scheduler_output, input_batch))
        self._step_state.layout = object()
        self._step_state.phase = VLLMStepPhase.LAYOUT_READY

    def _preflight_v2_force_eager(self, descriptor, **kwargs) -> bool:
        self.events.append(("preflight", descriptor, kwargs))
        self._step_state.capacity_candidate = (
            descriptor.cg_mode,
            descriptor,
        )
        return self.force_eager

    def _commit_actual_dispatch(
        self,
        scheduler_output,
        _model_runner,
        mode,
        descriptor,
        combined_force_eager,
    ) -> None:
        self.events.append(
            (
                "commit",
                scheduler_output,
                mode,
                descriptor,
                combined_force_eager,
            )
        )
        self._step_state.phase = VLLMStepPhase.COMMITTED


def _wrapped_worker(*, force_eager: bool = False):
    events: list[tuple] = []
    scheduler_output = _scheduler()
    input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["B", "A"],
        num_scheduled_tokens=np.array([3, 2], dtype=np.int32),
        num_computed_tokens_np=np.array([7, 11], dtype=np.int32),
        num_tokens=5,
    )

    def original_prepare(received, batch_req_state, descriptor):
        events.append(("prepare", received, batch_req_state, descriptor))
        return input_batch

    graph_candidate = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=4,
        uniform_token_count=1,
    )

    def original_dispatch(
        num_reqs,
        num_tokens,
        uniform_token_count,
        num_active_loras,
        max_query_len=None,
    ):
        events.append(
            (
                "dispatch",
                num_reqs,
                num_tokens,
                uniform_token_count,
                num_active_loras,
                max_query_len,
            )
        )
        return graph_candidate

    manager = SimpleNamespace(dispatch=original_dispatch)
    runner = SimpleNamespace(
        prepare_inputs=original_prepare,
        cudagraph_manager=manager,
    )
    adaptor = _FakeAdaptor(events, force_eager)
    worker = DMXV2GPUWorker.__new__(DMXV2GPUWorker)
    worker.adaptor = adaptor
    worker.model_runner = runner
    worker._dmx_v2_dispatch_manager = None
    worker._dmx_v2_original_dispatch = None
    worker._install_v2_prepare_wrapper()
    worker._ensure_v2_dispatch_wrapper()
    return worker, adaptor, runner, manager, scheduler_output, graph_candidate, events


def test_v2_wrappers_commit_after_real_dispatch_and_prepare() -> None:
    worker, adaptor, runner, manager, scheduler_output, candidate, events = (
        _wrapped_worker()
    )
    del worker
    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.ARMED,
        scheduler_output=scheduler_output,
    )

    descriptor = manager.dispatch(2, 5, 1, num_active_loras=0, max_query_len=3)
    assert descriptor is candidate
    batch_req_state = object()
    runner.prepare_inputs(scheduler_output, batch_req_state, descriptor)
    assert events[0][-1] == 3
    assert events[2][2] is batch_req_state

    assert [event[0] for event in events] == [
        "dispatch",
        "preflight",
        "prepare",
        "record",
        "commit",
    ]
    assert adaptor._step_state.phase is VLLMStepPhase.COMMITTED


def test_v2_wrapper_forces_eager_and_restores_dispatch() -> None:
    worker, adaptor, runner, manager, scheduler_output, candidate, events = (
        _wrapped_worker(force_eager=True)
    )
    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.ARMED,
        scheduler_output=scheduler_output,
    )

    descriptor = manager.dispatch(2, 5, 1, num_active_loras=0)
    assert descriptor == BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=5,
        num_reqs=2,
        num_active_loras=0,
    )
    runner.prepare_inputs(scheduler_output, object(), descriptor)
    assert adaptor._step_state.force_eager_latch

    worker._restore_v2_dispatch_wrapper()
    events.clear()
    assert manager.dispatch(2, 5, 1, 0) is candidate
    assert [event[0] for event in events] == ["dispatch"]


def test_v2_wrappers_preserve_pinned_signatures() -> None:
    _worker, _adaptor, runner, manager, _scheduler, _candidate, _events = (
        _wrapped_worker()
    )
    wrapped_prepare = list(
        inspect.signature(runner.prepare_inputs).parameters.values()
    )
    pinned_prepare = list(
        inspect.signature(GPUModelRunner.prepare_inputs).parameters.values()
    )[1:]
    wrapped_dispatch = list(
        inspect.signature(manager.dispatch).parameters.values()
    )
    pinned_dispatch = list(
        inspect.signature(CudaGraphManager.dispatch).parameters.values()
    )[1:]

    assert [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in wrapped_prepare
    ] == [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in pinned_prepare
    ]
    assert [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in wrapped_dispatch
    ] == [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in pinned_dispatch
    ]
