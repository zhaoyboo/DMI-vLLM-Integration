"""CPU contracts for GLM-5.2 decoder monitoring support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dmi_vllm_integration.architectures import ARCHITECTURE_REMAP
import dmi_vllm_integration.models.glm_moe_dsa as glm_moe_dsa_p
import tests.oracles.glm_moe_dsa_compare as glm_moe_dsa_compare
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2MoE,
    GlmMoeDsaForCausalLM,
)
from tests.oracles.glm_moe_dsa_compare import (
    GlmMoeDsaCompareForCausalLM,
)
from dmi_vllm_integration.models.glm_moe_dsa import (
    GlmMoeDsaPDecoderLayer,
    GlmMoeDsaPForCausalLM,
    GlmMoeDsaPMLP,
    GlmMoeDsaPModel,
    GlmMoeDsaPMoE,
    _instrument_glm52_model,
    _require_supported_glm52_config,
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
    HOOK_TYPE_MLP_POST,
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
from dmi_vllm_integration.model_shape import make_model_shape_from_hf_config

pytestmark = pytest.mark.vllm


def _indexer_types() -> list[str]:
    return [
        "full" if layer_no < 3 or (layer_no - 2) % 4 == 0 else "shared"
        for layer_no in range(78)
    ]


def _config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "glm_moe_dsa",
        "hidden_act": "silu",
        "hidden_size": 6144,
        "intermediate_size": 12_288,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "moe_router_dtype": "float32",
        "ep_size": 1,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "index_topk_pattern": None,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "indexer_rope_interleave": True,
        "index_share_for_mtp_iteration": True,
        "max_position_embeddings": 1_048_576,
        "vocab_size": 154_880,
        "rms_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "num_nextn_predict_layers": 1,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": False,
        "llama_4_scaling": None,
        "layer_types": ["deepseek_sparse_attention"] * 78,
        "mlp_layer_types": [*(["dense"] * 3), *(["sparse"] * 75)],
        "indexer_types": _indexer_types(),
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 8_000_000,
        },
        "rope_interleave": True,
        "quantization_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _parallel(**overrides) -> SimpleNamespace:
    values = {
        "tensor_parallel_size": 32,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "use_sequence_parallel_moe": False,
        "enable_eplb": False,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_glm52_preserves_upstream_class_and_loader_contracts() -> None:
    assert ARCHITECTURE_REMAP["GlmMoeDsaForCausalLM"] == "DMIGlmMoeDsaForCausalLM"
    assert issubclass(GlmMoeDsaPForCausalLM, GlmMoeDsaForCausalLM)
    assert issubclass(GlmMoeDsaCompareForCausalLM, GlmMoeDsaPForCausalLM)
    assert issubclass(GlmMoeDsaPModel, DeepseekV2Model)
    assert issubclass(GlmMoeDsaPDecoderLayer, DeepseekV2DecoderLayer)
    assert issubclass(GlmMoeDsaPMLP, DeepseekV2MLP)
    assert issubclass(GlmMoeDsaPMoE, DeepseekV2MoE)
    assert GlmMoeDsaPForCausalLM.load_weights is GlmMoeDsaForCausalLM.load_weights
    assert (
        GlmMoeDsaPForCausalLM.get_expert_mapping
        is GlmMoeDsaForCausalLM.get_expert_mapping
    )
    assert (
        GlmMoeDsaPForCausalLM.packed_modules_mapping
        == GlmMoeDsaForCausalLM.packed_modules_mapping
    )
    assert GlmMoeDsaPForCausalLM.model_cls is GlmMoeDsaPModel


def test_glm52_instruments_children_during_p_backbone_construction(
    monkeypatch,
) -> None:
    events = []

    def upstream_init(self, *, vllm_config, prefix="") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        events.append(("upstream", type(self), prefix))

    def instrument(model, *args, **kwargs):
        events.append(("instrument", type(model), args, kwargs))
        return model

    monkeypatch.setattr(DeepseekV2Model, "__init__", upstream_init)
    monkeypatch.setattr(glm_moe_dsa_p, "_instrument_glm52_model", instrument)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=_config(),
            dtype=torch.bfloat16,
            use_mla=True,
        ),
        parallel_config=_parallel(),
        quant_config=None,
        speculative_config=None,
    )

    model = GlmMoeDsaPModel(vllm_config=vllm_config, prefix="root.model")

    assert type(model) is GlmMoeDsaPModel
    assert events[0] == ("upstream", GlmMoeDsaPModel, "root.model")
    assert events[1][0:2] == ("instrument", GlmMoeDsaPModel)


def test_glm52_public_constructor_uses_p_backbone_factory(monkeypatch) -> None:
    observed = {}

    def model_init(self, *, vllm_config, prefix="") -> None:
        nn.Module.__init__(self)
        observed.update(model_type=type(self), prefix=prefix)

    def upstream_init(self, *, vllm_config, prefix="") -> None:
        nn.Module.__init__(self)
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model",
        )

    monkeypatch.setattr(GlmMoeDsaPModel, "__init__", model_init)
    monkeypatch.setattr(GlmMoeDsaForCausalLM, "__init__", upstream_init)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=_config(),
            dtype=torch.bfloat16,
            use_mla=True,
        ),
        parallel_config=_parallel(),
        quant_config=None,
        speculative_config=None,
    )

    subject = GlmMoeDsaPForCausalLM(
        vllm_config=vllm_config,
        prefix="root",
    )

    assert type(subject.model) is GlmMoeDsaPModel
    assert observed == {
        "model_type": GlmMoeDsaPModel,
        "prefix": "root.model",
    }


def test_glm52_instrumenter_rejects_late_root_class_swapping() -> None:
    upstream_model = DeepseekV2Model.__new__(DeepseekV2Model)
    nn.Module.__init__(upstream_model)

    with pytest.raises(TypeError, match="backbone must be constructed"):
        _instrument_glm52_model(upstream_model)


@pytest.mark.parametrize(
    "config_overrides,parallel_overrides,dtype,speculative_config",
    [
        ({"hidden_size": 7168}, {}, None, None),
        ({"first_k_dense_replace": 1}, {}, None, None),
        ({"layer_types": ["full_attention"] * 78}, {}, None, None),
        ({"quantization_config": {"quant_method": "fp8"}}, {}, None, None),
        ({}, {"tensor_parallel_size": 16}, None, None),
        ({}, {"pipeline_parallel_size": 2}, None, None),
        ({}, {"enable_expert_parallel": True}, None, None),
        ({}, {"use_sequence_parallel_moe": True}, None, None),
        ({}, {"enable_eplb": True}, None, None),
        ({}, {}, torch.bfloat16, object()),
        ({}, {}, torch.float32, None),
    ],
)
def test_glm52_does_not_reject_upstream_owned_or_shared_runtime_options(
    config_overrides,
    parallel_overrides,
    dtype,
    speculative_config,
) -> None:
    _require_supported_glm52_config(
        _config(**config_overrides),
        _parallel(**parallel_overrides),
        quant_config=object(),
        dtype=dtype,
        speculative_config=speculative_config,
    )


def test_glm52_rejects_unimplemented_decoder_branches() -> None:
    with pytest.raises(NotImplementedError, match="requires vLLM MLA"):
        _require_supported_glm52_config(
            _config(),
            _parallel(),
            use_mla=False,
        )
    with pytest.raises(NotImplementedError, match="FP16"):
        _require_supported_glm52_config(
            _config(),
            _parallel(),
            dtype=torch.float16,
        )
    with pytest.raises(NotImplementedError, match="llama_4_scaling=None"):
        _require_supported_glm52_config(
            _config(llama_4_scaling={"beta": 0.1}),
            _parallel(),
        )


def test_glm52_model_shape_reads_deepseek_style_expert_count() -> None:
    shape = make_model_shape_from_hf_config(_config(), torch.bfloat16)

    assert shape is not None
    assert shape.hidden_dim == 6144
    assert shape.intermediate_dim == 12_288
    assert shape.num_experts == 256
    assert shape.top_k == 8


def test_glm52_dense_mlp_post_hook_observes_post_activation_values() -> None:
    subject = GlmMoeDsaPMLP.__new__(GlmMoeDsaPMLP)
    nn.Module.__init__(subject)

    class GateUp(nn.Module):
        def forward(self, value):
            return value + 2, None

    class Activation(nn.Module):
        def forward(self, value):
            return value * 3

    class Down(nn.Module):
        def forward(self, value):
            return value - 5, None

    subject.gate_up_proj = GateUp()
    subject.act_fn = Activation()
    subject.down_proj = Down()
    subject.hook_post = HookPoint()
    captured = []
    subject.hook_post.register_forward_hook(
        lambda _module, _args, output: captured.append(output.clone())
    )
    inputs = torch.tensor([[1.0, 2.0]])

    output = subject(inputs)

    expected_post = (inputs + 2) * 3
    assert torch.equal(captured[0], expected_post)
    assert torch.equal(output, expected_post - 5)


def test_glm52_routing_hooks_preserve_runner_owned_gate() -> None:
    subject = GlmMoeDsaPMoE.__new__(GlmMoeDsaPMoE)
    nn.Module.__init__(subject)
    subject.is_sequence_parallel = False
    events = []

    class Gate(nn.Module):
        def forward(self, hidden_states):
            events.append("gate")
            return (
                torch.arange(
                    hidden_states.shape[0] * 4,
                    dtype=torch.float32,
                ).reshape(hidden_states.shape[0], 4),
                None,
            )

    class Router:
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
            self.router = Router()

        def forward(self, *, hidden_states, router_logits):
            assert router_logits is hidden_states
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
        hook.register_forward_hook(
            lambda _module, _args, output, name=name: captured.setdefault(
                name,
                output.clone(),
            )
        )
        setattr(subject, f"hook_{name}", hook)
    subject.experts.router.set_routing_observer(subject._observe_routing)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    gate_logits = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    output = subject(hidden_states)

    assert events == ["gate", "select", "experts"]
    assert torch.equal(output, hidden_states + 1)
    assert captured["router_logits"].dtype == torch.float32
    assert captured["topk_ids"].dtype == torch.int32
    assert captured["topk_weights"].dtype == torch.float32
    assert subject.experts.router.calls == 1


def test_glm52_compare_buffers_cover_dense_and_sparse_families(monkeypatch) -> None:
    subject = GlmMoeDsaCompareForCausalLM.__new__(GlmMoeDsaCompareForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config(num_hidden_layers=4)
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=4,
        layers=[SimpleNamespace(mlp=SimpleNamespace()) for _ in range(4)],
    )
    allocations = []

    def fake_empty(*shape, **kwargs):
        value = SimpleNamespace(shape=shape, **kwargs)
        allocations.append(value)
        return value

    monkeypatch.setattr(glm_moe_dsa_compare.torch, "empty", fake_empty)
    monkeypatch.setattr(
        glm_moe_dsa_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 32,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )

    subject.allocate_compare_buffers(16, vllm_config)
    buffers = subject.get_ref_buffers()

    assert len(allocations) == 39
    assert len(buffers) == 39
    assert buffers["mlp_post_L0"].shape == (16, 384)
    assert "router_logits_L2" not in buffers
    assert buffers["router_logits_L3"].shape == (16, 256)
    assert buffers["router_logits_L3"].dtype == torch.float32
    assert buffers["topk_ids_L3"].shape == (16, 8)
    assert buffers["topk_ids_L3"].dtype == torch.int32
    assert buffers["topk_weights_L3"].dtype == torch.float32
    assert buffers["final_logits"].shape == (4, 154_880)


@pytest.mark.parametrize(
    ("subject_cls", "upstream_cls", "hook_names", "args", "expected_args"),
    [
        (
            GlmMoeDsaPMLP,
            DeepseekV2MLP,
            ("post",),
            (torch.tensor([[1.0]]),),
            (torch.tensor([[1.0]]),),
        ),
        (
            GlmMoeDsaPMoE,
            DeepseekV2MoE,
            ("router_logits", "topk_ids", "topk_weights"),
            (torch.tensor([[1.0]]),),
            (torch.tensor([[1.0]]), False),
        ),
        (
            GlmMoeDsaPDecoderLayer,
            DeepseekV2DecoderLayer,
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
            (torch.tensor([0]), torch.tensor([[1.0]]), None, None),
        ),
    ],
)
def test_glm52_disabled_hooks_delegate_to_upstream(
    monkeypatch,
    subject_cls,
    upstream_cls,
    hook_names,
    args,
    expected_args,
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
    assert len(observed) == 1
    assert len(observed[0]) == len(expected_args)
    for actual, expected in zip(observed[0], expected_args, strict=True):
        if isinstance(actual, torch.Tensor):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected


def test_glm52_model_wide_manifest_has_779_truthful_families() -> None:
    subject = GlmMoeDsaPForCausalLM.__new__(GlmMoeDsaPForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config()
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=78,
        layers=[None] * 78,
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    common_prefix = [
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
    ]
    expected = [HOOK_TYPE_TOKEN_IDS, HOOK_TYPE_EMBED]
    for layer_no in range(78):
        expected.extend(common_prefix)
        if layer_no < 3:
            expected.append(HOOK_TYPE_MLP_POST)
        else:
            expected.extend(
                [
                    HOOK_TYPE_ROUTER_LOGITS,
                    HOOK_TYPE_TOPK_IDS,
                    HOOK_TYPE_TOPK_WEIGHTS,
                ]
            )
        expected.append(HOOK_TYPE_MLP_OUT)
    expected.extend([HOOK_TYPE_RESID_FINAL, HOOK_TYPE_FINAL_LN, HOOK_TYPE_FINAL_LOGITS])

    specs = subject.get_hook_specs(model_wide=True)

    assert len(specs) == 779
    assert [spec.hook_type for spec in specs] == expected
    assert all(spec.module is None for spec in specs)
    assert sum(spec.hook_type == HOOK_TYPE_MLP_POST for spec in specs) == 3
    assert sum(spec.hook_type == HOOK_TYPE_ROUTER_LOGITS for spec in specs) == 75
    assert not {HOOK_TYPE_Q, HOOK_TYPE_K, HOOK_TYPE_V, HOOK_TYPE_Z}.intersection(
        spec.hook_type for spec in specs
    )


def test_glm52_sequence_parallel_marker_covers_each_custom_forward() -> None:
    assert GlmMoeDsaPForCausalLM.dmi_sequence_parallel_unsupported_hook_types == {
        HOOK_TYPE_EMBED,
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_OUT,
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
    }
