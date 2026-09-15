"""CPU contracts for dense Granite monitoring support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dmi_vllm_integration.architectures import ARCHITECTURE_REMAP
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
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
)
from vllm.model_executor.models.granite import (
    GraniteAttention,
    GraniteDecoderLayer,
    GraniteForCausalLM,
    GraniteMLP,
    GraniteModel,
)
import dmi_vllm_integration.models.granite as granite_p_module
import tests.oracles.granite_compare as granite_compare
from dmi_vllm_integration.models.granite import (
    GranitePAttention,
    GranitePDecoderLayer,
    GranitePForCausalLM,
    GranitePMLP,
    GranitePModel,
    _instrument_granite_model,
)


pytestmark = pytest.mark.vllm


def _config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "granite",
        "hidden_act": "silu",
        "attention_multiplier": 0.015625,
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22,
        "logits_scaling": 10.0,
        "num_hidden_layers": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_granite_preserves_upstream_loader_and_class_contracts() -> None:
    assert ARCHITECTURE_REMAP["GraniteForCausalLM"] == "DMIGraniteForCausalLM"
    assert issubclass(GranitePForCausalLM, GraniteForCausalLM)
    assert issubclass(GranitePModel, GraniteModel)
    assert issubclass(GranitePDecoderLayer, GraniteDecoderLayer)
    assert issubclass(GranitePAttention, GraniteAttention)
    assert issubclass(GranitePMLP, GraniteMLP)
    assert (
        GranitePForCausalLM.packed_modules_mapping
        == GraniteForCausalLM.packed_modules_mapping
    )
    assert (
        GranitePForCausalLM.hf_to_vllm_mapper
        is GraniteForCausalLM.hf_to_vllm_mapper
    )


@pytest.mark.parametrize(
    ("logits_scaling", "expected_scale"),
    [(None, 2.0), (10.0, 0.2)],
)
def test_granite_mirrors_upstream_optional_logits_scaling(
    monkeypatch,
    logits_scaling,
    expected_scale,
) -> None:
    config_values = {
        "vocab_size": 32,
        "hidden_size": 8,
        "tie_word_embeddings": False,
        "logit_scale": 2.0,
    }
    if logits_scaling is not None:
        config_values["logits_scaling"] = logits_scaling
    config = SimpleNamespace(**config_values)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
    )

    class FakeModel(nn.Module):
        def __init__(self, **_kwargs) -> None:
            super().__init__()

    class FakeHead(nn.Module):
        pass

    class FakeLogitsProcessor(nn.Module):
        def __init__(self, _vocab_size, *, scale) -> None:
            super().__init__()
            self.scale = scale

    monkeypatch.setattr(
        granite_p_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )
    monkeypatch.setattr(
        granite_p_module,
        "ParallelLMHead",
        lambda *_args, **_kwargs: FakeHead(),
    )
    monkeypatch.setattr(granite_p_module, "LogitsProcessor", FakeLogitsProcessor)

    subject = GranitePForCausalLM(
        vllm_config=vllm_config,
        model_type=FakeModel,
    )

    assert subject.logits_processor.scale == expected_scale


def test_granite_disabled_layer_hooks_delegate_to_upstream(monkeypatch) -> None:
    layer = GranitePDecoderLayer.__new__(GranitePDecoderLayer)
    nn.Module.__init__(layer)
    for name in (
        "resid_pre",
        "ln1",
        "attn_out",
        "resid_mid",
        "ln2",
        "mlp_in",
        "mlp_out",
    ):
        hook = HookPoint()
        hook.enabled = False
        setattr(layer, f"hook_{name}", hook)
    marker = torch.tensor([[17.0]])
    observed = []

    def upstream_forward(_self, positions, hidden_states):
        observed.append((positions, hidden_states))
        return marker

    monkeypatch.setattr(GraniteDecoderLayer, "forward", upstream_forward)
    positions = torch.tensor([0])
    hidden_states = torch.tensor([[1.0]])

    result = layer(positions, hidden_states)

    assert result is marker
    assert observed == [(positions, hidden_states)]


class _Scale(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.amount


class _Attention(nn.Module):
    def forward(self, *, positions, hidden_states):
        del positions
        return hidden_states + 3


class _Add(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.amount


def test_granite_layer_hooks_preserve_scaled_residual_order() -> None:
    layer = GranitePDecoderLayer.__new__(GranitePDecoderLayer)
    nn.Module.__init__(layer)
    layer.residual_multiplier = 0.25
    layer.input_layernorm = _Scale(2)
    layer.self_attn = _Attention()
    layer.post_attention_layernorm = _Scale(2)
    layer.mlp = _Add(5)
    captured: dict[str, torch.Tensor] = {}
    for name in (
        "resid_pre",
        "ln1",
        "attn_out",
        "resid_mid",
        "ln2",
        "mlp_in",
        "mlp_out",
    ):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, key=name: captured.setdefault(
                key,
                output.clone(),
            )
        )
        setattr(layer, f"hook_{name}", hook)

    output = layer(torch.tensor([0]), torch.tensor([[1.0]]))

    assert captured["resid_pre"].item() == 1.0
    assert captured["ln1"].item() == 2.0
    assert captured["attn_out"].item() == 5.0
    assert captured["resid_mid"].item() == 2.25
    assert captured["ln2"].item() == 4.5
    assert captured["mlp_in"].item() == 4.5
    assert captured["mlp_out"].item() == 9.5
    assert output.item() == 4.625


class _Projection(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, _value: torch.Tensor):
        return self.output, None


class _Rotary(nn.Module):
    def forward(self, _positions, q, k):
        return q + 100, k + 100


class _AttentionKernel(nn.Module):
    def forward(self, q, _k, _v):
        return q


@pytest.mark.parametrize("use_rope", [True, False])
def test_granite_attention_hooks_bind_pre_rope_qkv_and_pre_o_proj_z(use_rope) -> None:
    attention = GranitePAttention.__new__(GranitePAttention)
    nn.Module.__init__(attention)
    attention.q_size = 4
    attention.kv_size = 2
    attention.num_heads = 2
    attention.num_kv_heads = 1
    attention.head_dim = 2
    qkv = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    attention.qkv_proj = _Projection(qkv)
    attention.rotary_emb = _Rotary()
    attention.use_rope = use_rope
    attention.attn = _AttentionKernel()
    attention.o_proj = _Projection(torch.full((1, 4), 99.0))
    captured: dict[str, torch.Tensor] = {}
    for name in ("q", "k", "v", "z"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, key=name: captured.setdefault(
                key,
                output.clone(),
            )
        )
        setattr(attention, f"hook_{name}", hook)

    output = attention(torch.tensor([0]), torch.zeros(1, 4))

    assert output.tolist() == [[99.0, 99.0, 99.0, 99.0]]
    assert torch.equal(captured["q"], qkv[:, :4].view(1, 2, 2))
    assert torch.equal(captured["k"], qkv[:, 4:6].view(1, 1, 2))
    assert torch.equal(captured["v"], qkv[:, 6:].view(1, 1, 2))
    assert torch.equal(captured["z"], qkv[:, :4] + (100 if use_rope else 0))


def test_granite_instruments_the_upstream_tree_in_place() -> None:
    model = GraniteModel.__new__(GraniteModel)
    nn.Module.__init__(model)
    model.config = _config()
    model.start_layer = 0
    model.end_layer = 1
    layer = GraniteDecoderLayer.__new__(GraniteDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = GraniteAttention.__new__(GraniteAttention)
    nn.Module.__init__(layer.self_attn)
    layer.mlp = GraniteMLP.__new__(GraniteMLP)
    nn.Module.__init__(layer.mlp)
    model.layers = nn.ModuleList([layer])
    identities = (id(model), id(layer), id(layer.self_attn), id(layer.mlp))

    result = _instrument_granite_model(model)

    assert (id(result), id(layer), id(layer.self_attn), id(layer.mlp)) == identities
    assert isinstance(result, GranitePModel)
    assert isinstance(layer, GranitePDecoderLayer)
    assert isinstance(layer.self_attn, GranitePAttention)
    assert isinstance(layer.mlp, GranitePMLP)


def _fake_hooked_model() -> GranitePForCausalLM:
    subject = GranitePForCausalLM.__new__(GranitePForCausalLM)
    nn.Module.__init__(subject)
    attention = SimpleNamespace(
        **{f"hook_{name}": HookPoint() for name in ("q", "k", "v", "z")}
    )
    mlp = SimpleNamespace(hook_post=HookPoint())
    layer = SimpleNamespace(
        self_attn=attention,
        mlp=mlp,
        **{
            f"hook_{name}": HookPoint()
            for name in (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            )
        },
    )
    subject.config = _config()
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=1,
        layers=[layer],
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    return subject


def test_granite_manifest_is_truthful_and_in_firing_order() -> None:
    specs = _fake_hooked_model().get_hook_specs()

    assert [spec.hook_type for spec in specs] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
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
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_MLP_OUT,
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    assert all(spec.module is not None for spec in specs)


def test_granite_pp_manifests_separate_local_and_model_wide_layers() -> None:
    subject = _fake_hooked_model()
    local_layer = subject.model.layers[0]
    subject.config.num_hidden_layers = 3
    subject.model.start_layer = 1
    subject.model.end_layer = 2
    subject.model.layers = [None, local_layer, None]

    local_specs = subject.get_hook_specs()
    model_wide_specs = subject.get_hook_specs(model_wide=True)

    assert len(local_specs) == 17
    assert {
        spec.layer_no for spec in local_specs if spec.layer_no >= 0
    } == {1}
    assert all(spec.module is not None for spec in local_specs)
    assert len(model_wide_specs) == 41
    assert all(spec.module is None for spec in model_wide_specs)


def test_granite_compare_uses_the_constructed_mlp_width(monkeypatch) -> None:
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            down_proj=SimpleNamespace(input_size_per_partition=256)
        ),
        self_attn=SimpleNamespace(),
    )
    model = SimpleNamespace(start_layer=0, end_layer=1, layers=[layer])
    subject = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=128,
            vocab_size=65_536,
        ),
        model=model,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            head_dtype=torch.float32,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    allocations = []

    def fake_empty(*shape, **kwargs):
        allocations.append((shape, kwargs))
        return shape

    monkeypatch.setattr(
        granite_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        granite_compare.torch,
        "empty",
        fake_empty,
    )

    granite_compare.GraniteCompareForCausalLM.allocate_compare_buffers(
        subject,
        8,
        vllm_config,
    )

    assert layer.mlp._buf_mlp_post == (8, 256)
    assert any(
        shape == (4, 65_536) and kwargs["dtype"] is torch.float32
        for shape, kwargs in allocations
    )


def test_granite_compare_handles_pipeline_stage_without_logits() -> None:
    subject = SimpleNamespace(
        logits_processor=lambda *_args: None,
        lm_head=object(),
    )

    assert (
        granite_compare.GraniteCompareForCausalLM.compute_logits(
            subject,
            torch.zeros(1, 1),
        )
        is None
    )


def test_granite_compare_constructs_concrete_compiled_backbone(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_init(
        self,
        *,
        vllm_config,
        prefix,
        model_type=GranitePModel,
    ) -> None:
        nn.Module.__init__(self)
        observed.update(
            vllm_config=vllm_config,
            prefix=prefix,
            model_type=model_type,
        )

    monkeypatch.setattr(GranitePForCausalLM, "__init__", fake_init)
    config = object()

    granite_compare.GraniteCompareForCausalLM(
        vllm_config=config,
        prefix="model",
    )

    assert observed == {
        "vllm_config": config,
        "prefix": "model",
        "model_type": granite_compare.GraniteCompareModel,
    }
