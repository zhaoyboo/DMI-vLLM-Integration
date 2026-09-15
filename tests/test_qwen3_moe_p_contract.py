"""CPU contracts for Qwen3-MoE monitoring support."""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode

from dmi_vllm_integration.architectures import ARCHITECTURE_REMAP
import tests.oracles.qwen3_moe_compare as qwen3_moe_compare
import vllm.model_executor.models.qwen3_moe as upstream_qwen3_moe
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from tests.oracles.qwen3_moe_compare import (
    Qwen3MoeCompareForCausalLM,
)
from dmi_vllm_integration.models.qwen3_moe import (
    Qwen3MoePAttention,
    Qwen3MoePDecoderLayer,
    Qwen3MoePForCausalLM,
    Qwen3MoePModel,
    Qwen3MoePSparseMoeBlock,
    _instrument_qwen3_moe_model,
    _require_supported_qwen3_moe_config,
)

from dmi_vllm_integration.dmi_api import HookPoint
from dmi_vllm_integration.dmi_api import (
    HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_EMBED,
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_K,
    HOOK_TYPE_LN1,
    HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
)

pytestmark = pytest.mark.vllm


def _config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "qwen3_moe",
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "output_router_logits": False,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "shared_expert_intermediate_size": None,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
        "vocab_size": 151_936,
        "max_position_embeddings": 40_960,
        "rms_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "rope_scaling": None,
        "rope_theta": 1_000_000,
        "quantization_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _parallel(**overrides) -> SimpleNamespace:
    values = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "use_sequence_parallel_moe": False,
        "enable_eplb": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen3_moe_preserves_upstream_class_and_loader_contracts() -> None:
    assert ARCHITECTURE_REMAP["Qwen3MoeForCausalLM"] == "DMIQwen3MoeForCausalLM"
    assert issubclass(Qwen3MoePForCausalLM, Qwen3MoeForCausalLM)
    assert issubclass(Qwen3MoeCompareForCausalLM, Qwen3MoePForCausalLM)
    assert issubclass(Qwen3MoePModel, Qwen3MoeModel)
    assert issubclass(Qwen3MoePDecoderLayer, Qwen3MoeDecoderLayer)
    assert issubclass(Qwen3MoePAttention, Qwen3MoeAttention)
    assert issubclass(Qwen3MoePSparseMoeBlock, Qwen3MoeSparseMoeBlock)
    assert (
        Qwen3MoePForCausalLM.packed_modules_mapping
        == Qwen3MoeForCausalLM.packed_modules_mapping
    )
    assert vars(Qwen3MoePForCausalLM.hf_to_vllm_mapper) == vars(
        Qwen3MoeForCausalLM.hf_to_vllm_mapper
    )
    assert Qwen3MoePForCausalLM.fall_back_to_pt_during_load is False


def test_qwen3_moe_constructs_dmi_backbone_before_compile_wrapper() -> None:
    params = signature(Qwen3MoePForCausalLM.__init__).parameters

    assert params["model_type"].default is Qwen3MoePModel


def test_qwen3_moe_constructor_uses_backbone_factory() -> None:
    observed = {}

    class ConstructionStopped(Exception):
        pass

    def model_type(*, vllm_config, prefix):
        observed.update(vllm_config=vllm_config, prefix=prefix)
        raise ConstructionStopped

    config = _config()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=config,
            dtype=torch.bfloat16,
        ),
        parallel_config=_parallel(),
        quant_config=None,
    )

    with pytest.raises(ConstructionStopped):
        Qwen3MoePForCausalLM(
            vllm_config=vllm_config,
            prefix="root",
            model_type=model_type,
        )

    assert observed == {
        "vllm_config": vllm_config,
        "prefix": "root.model",
    }


def test_qwen3_moe_compile_wrapper_captures_dmi_backbone_forward(
    monkeypatch,
) -> None:
    captured = []
    monkeypatch.setattr(compilation_counter, "num_models_seen", 0)
    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper,
        "__init__",
        lambda self, **_kwargs: captured.append(self.forward.__func__),
    )
    monkeypatch.setattr(
        upstream_qwen3_moe,
        "VocabParallelEmbedding",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        upstream_qwen3_moe,
        "make_layers",
        lambda *_args, **_kwargs: (0, 0, nn.ModuleList()),
    )
    monkeypatch.setattr(
        upstream_qwen3_moe,
        "make_empty_intermediate_tensors_factory",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        upstream_qwen3_moe,
        "RMSNorm",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    config = _config(num_hidden_layers=0)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        quant_config=None,
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.VLLM_COMPILE),
    )

    Qwen3MoePModel(vllm_config=vllm_config)

    assert captured == [Qwen3MoePModel.forward]


def test_qwen3_moe_instrumenter_rejects_late_class_swapping() -> None:
    upstream_model = SimpleNamespace(config=_config())

    with pytest.raises(TypeError, match="constructed as its DMI class"):
        _instrument_qwen3_moe_model(upstream_model, _parallel())


@pytest.mark.parametrize(
    "overrides",
    [
        {"decoder_sparse_step": 2},
        {"mlp_only_layers": [0]},
    ],
)
def test_qwen3_moe_requires_every_decoder_layer_to_be_moe(overrides) -> None:
    with pytest.raises(NotImplementedError, match="every decoder layer"):
        _require_supported_qwen3_moe_config(_config(**overrides), _parallel())


@pytest.mark.parametrize(
    "config_overrides,parallel_overrides,dtype",
    [
        ({"hidden_size": 4096}, {}, None),
        ({"rope_scaling": {"rope_type": "yarn"}}, {}, None),
        ({"quantization_config": {"quant_method": "gptq"}}, {}, None),
        ({"shared_expert_intermediate_size": 512}, {}, None),
        ({}, {"tensor_parallel_size": 2}, None),
        ({}, {"pipeline_parallel_size": 2}, None),
        ({}, {"enable_expert_parallel": True}, None),
        ({}, {"use_sequence_parallel_moe": True}, None),
        ({}, {"enable_eplb": True}, None),
        ({}, {}, torch.float16),
    ],
)
def test_qwen3_moe_does_not_reject_upstream_owned_or_shared_runtime_options(
    config_overrides,
    parallel_overrides,
    dtype,
) -> None:
    _require_supported_qwen3_moe_config(
        _config(**config_overrides),
        _parallel(**parallel_overrides),
        dtype=dtype,
    )


def test_qwen3_moe_routing_hooks_observe_authoritative_gate_values() -> None:
    subject = Qwen3MoePSparseMoeBlock.__new__(Qwen3MoePSparseMoeBlock)
    nn.Module.__init__(subject)
    subject.is_sequence_parallel = False
    events = []

    class Gate(nn.Module):
        def forward(self, hidden_states):
            events.append("gate")
            logits = torch.arange(
                hidden_states.shape[0] * 4,
                dtype=torch.float32,
            ).reshape(hidden_states.shape[0], 4)
            return logits, None

    class ExpertRouter:
        observer = None
        calls = 0

        def set_routing_observer(self, observer):
            self.observer = observer

        def select_experts(self, *, hidden_states, router_logits):
            self.calls += 1
            events.append("select")
            assert hidden_states.shape == (2, 4)
            assert router_logits.shape == (2, 4)
            route = (
                torch.full((2, 2), 0.5, dtype=torch.float32),
                torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            )
            if self.observer is not None:
                self.observer(router_logits, *route)
            return route

    class Experts(nn.Module):

        def __init__(self):
            super().__init__()
            self.router = ExpertRouter()

        def forward(self, *, hidden_states, router_logits):
            assert router_logits is hidden_states
            router_logits, _ = subject.gate(hidden_states)
            self.used_route = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            events.append("experts")
            assert router_logits.shape == (2, 4)
            return hidden_states + 1

    subject.gate = Gate()
    subject.experts = Experts()
    captured = {}
    for name in ("router_logits", "topk_ids", "topk_weights"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, name=name: captured.setdefault(
                name,
                output,
            )
        )
        setattr(subject, f"hook_{name}", hook)
    subject.experts.router.set_routing_observer(subject._observe_routing)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    output = subject(hidden_states)

    assert events == ["gate", "select", "experts"]
    assert torch.equal(output, hidden_states + 1)
    assert captured["router_logits"].shape == (2, 4)
    assert captured["topk_ids"].dtype == torch.int32
    assert captured["topk_weights"].dtype == torch.float32
    assert subject.experts.router.calls == 1


def test_qwen3_moe_internal_router_preserves_upstream_runner_input() -> None:
    subject = Qwen3MoePSparseMoeBlock.__new__(Qwen3MoePSparseMoeBlock)
    nn.Module.__init__(subject)
    subject.is_sequence_parallel = False
    events = []

    class Gate(nn.Module):
        def forward(self, hidden_states):
            events.append("gate")
            return hidden_states + 20, None

    class ExpertRouter:
        observer = None
        calls = 0

        def set_routing_observer(self, observer):
            self.observer = observer

        def select_experts(self, *, hidden_states, router_logits):
            self.calls += 1
            events.append("select")
            route = (
                torch.ones((1, 1), dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.int32),
            )
            if self.observer is not None:
                self.observer(router_logits, *route)
            return route

    class Experts(nn.Module):

        def __init__(self):
            super().__init__()
            self.router = ExpertRouter()

        def forward(self, *, hidden_states, router_logits):
            assert torch.equal(router_logits, hidden_states)
            actual_logits, _ = subject.gate(hidden_states)
            self.used_route = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=actual_logits,
            )
            events.append("experts")
            return hidden_states + 1

    subject.gate = Gate()
    subject.experts = Experts()
    captured = {}
    for name in ("router_logits", "topk_ids", "topk_weights"):
        hook = HookPoint()
        hook.enabled = name == "router_logits"
        hook.register_forward_hook(
            lambda _module, _args, output, name=name: captured.setdefault(
                name,
                output,
            )
        )
        setattr(subject, f"hook_{name}", hook)
    subject.experts.router.set_routing_observer(subject._observe_routing)
    hidden_states = torch.arange(4, dtype=torch.float32).reshape(1, 4)

    output = subject(hidden_states)

    assert torch.equal(output, hidden_states + 1)
    assert events == ["gate", "select", "experts"]
    assert torch.equal(captured["router_logits"], hidden_states + 20)
    assert subject.experts.router.calls == 1


def test_qwen3_moe_compare_buffers_cover_every_manifest_family(
    monkeypatch,
) -> None:
    subject = Qwen3MoeCompareForCausalLM.__new__(Qwen3MoeCompareForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config(num_hidden_layers=2)
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=2,
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(),
                mlp=SimpleNamespace(),
            )
            for _ in range(2)
        ],
    )
    allocations = []

    def fake_empty(*shape, **kwargs):
        value = SimpleNamespace(shape=shape, **kwargs)
        allocations.append(value)
        return value

    monkeypatch.setattr(qwen3_moe_compare.torch, "empty", fake_empty)
    monkeypatch.setattr(
        qwen3_moe_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )

    subject.allocate_compare_buffers(16, vllm_config)
    buffers = subject.get_ref_buffers()

    assert len(allocations) == 33
    assert len(buffers) == 33
    assert buffers["q_L0"].shape == (16, 16, 128)
    assert buffers["k_L0"].shape == (16, 2, 128)
    assert buffers["router_logits_L0"].shape == (16, 128)
    assert buffers["topk_ids_L0"].dtype == torch.int32
    assert buffers["topk_weights_L0"].dtype == torch.float32
    assert buffers["final_logits"].shape == (4, 151_936)


@pytest.mark.parametrize(
    ("subject_cls", "upstream_cls", "hook_names", "args"),
    [
        (
            Qwen3MoePAttention,
            Qwen3MoeAttention,
            ("q", "k", "v", "z"),
            (torch.tensor([0]), torch.tensor([[1.0]])),
        ),
        (
            Qwen3MoePSparseMoeBlock,
            Qwen3MoeSparseMoeBlock,
            ("router_logits", "topk_ids", "topk_weights"),
            (torch.tensor([[1.0]]),),
        ),
        (
            Qwen3MoePDecoderLayer,
            Qwen3MoeDecoderLayer,
            (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ),
            (torch.tensor([0]), torch.tensor([[1.0]]), None),
        ),
    ],
)
def test_qwen3_moe_disabled_hooks_delegate_to_upstream(
    monkeypatch,
    subject_cls,
    upstream_cls,
    hook_names,
    args,
) -> None:
    subject = subject_cls.__new__(subject_cls)
    nn.Module.__init__(subject)
    for name in hook_names:
        hook = HookPoint()
        hook.enabled = False
        setattr(subject, f"hook_{name}", hook)
    marker = torch.tensor([[17.0]])
    observed = []

    def upstream_forward(_self, *forward_args):
        observed.append(forward_args)
        return marker

    monkeypatch.setattr(upstream_cls, "forward", upstream_forward)

    result = subject(*args)

    assert result is marker
    assert observed == [args]


def test_qwen3_moe_model_wide_manifest_has_677_truthful_families() -> None:
    subject = Qwen3MoePForCausalLM.__new__(Qwen3MoePForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config()
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=48,
        layers=[None] * 48,
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    layer_types = [
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_ROUTER_LOGITS,
        HOOK_TYPE_TOPK_IDS,
        HOOK_TYPE_TOPK_WEIGHTS,
        HOOK_TYPE_MLP_OUT,
    ]

    specs = subject.get_hook_specs(model_wide=True)

    assert len(specs) == 677
    assert [spec.hook_type for spec in specs] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        *(layer_types * 48),
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    assert all(spec.module is None for spec in specs)
    assert specs[13].dtype == torch.int32
    assert specs[14].dtype == torch.float32
