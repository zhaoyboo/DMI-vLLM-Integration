"""Focused contracts for the model ports required by vLLM 0.29.0."""

from __future__ import annotations

import ast
from collections.abc import Iterable
import importlib
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.vllm


GPT2_MODULES = (
    "dmi_vllm_integration.models.gpt2",
    "tests.oracles.gpt2_compare",
    "tests.oracles.gpt2_ref",
)
QWEN3_MODULES = (
    "dmi_vllm_integration.models.qwen3",
    "tests.oracles.qwen3_compare",
    "tests.oracles.qwen3_ref",
)


def _require_unmodified_router() -> None:
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter,
    )

    if hasattr(FusedMoERouter, "set_routing_observer"):
        pytest.skip("requires the unmodified official vLLM router")


def test_dense_qwen2_uses_the_registered_monitored_variant() -> None:
    from dmi_vllm_integration.adapter import _ARCH_REMAP
    from dmi_vllm_integration.models.qwen2 import Qwen2PForCausalLM

    monitored_arch = _ARCH_REMAP["Qwen2ForCausalLM"]
    assert monitored_arch == "DMIQwen2ForCausalLM"
    assert callable(Qwen2PForCausalLM.get_hook_specs)


@pytest.mark.parametrize("module_name", GPT2_MODULES)
def test_gpt2_variants_use_v029_auto_loader_contract(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(module_name)
    calls: dict[str, object] = {}

    class FakeLoader:
        def __init__(self, model: object) -> None:
            calls["model"] = model

        def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
            *, mapper,
        ) -> set[str]:
            calls["mapper"] = mapper
            materialized = list(weights)
            calls["weights"] = materialized
            return {name for name, _ in materialized}

    monkeypatch.setattr(module, "AutoWeightsLoader", FakeLoader)
    model = module.GPT2Model.__new__(module.GPT2Model)
    weight = torch.arange(6).reshape(2, 3)

    loaded = model.load_weights([("h.0.attn.c_proj.weight", weight)])

    assert calls["model"] is model
    assert calls["mapper"].orig_to_new_substr == {
        ".attn.bias": None, ".attn.masked_bias": None,
    }
    assert loaded == {"h.0.attn.c_proj.weight"}
    assert torch.equal(calls["weights"][0][1], weight.t())  # type: ignore[index]


@pytest.mark.parametrize("head_dtype", (torch.float16, torch.float32))
def test_gpt2_ref_final_logits_buffer_uses_head_dtype(
    head_dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = importlib.import_module("tests.oracles.gpt2_ref")
    ref_config = tmp_path / "ref_config.json"
    ref_config.write_text(
        '{"enabled_hooks": ["final_logits"], "max_len": 4}'
    )
    monkeypatch.setenv("REF_CONFIG", str(ref_config))

    parallel_state = importlib.import_module(
        "vllm.distributed.parallel_state"
    )
    monkeypatch.setattr(
        parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        module.torch,
        "empty",
        lambda *shape, dtype, device: SimpleNamespace(
            shape=shape, dtype=dtype, device=device
        ),
    )

    subject = SimpleNamespace(
        transformer=SimpleNamespace(start_layer=0, end_layer=0, h=[])
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float16,
            head_dtype=head_dtype,
            hf_config=SimpleNamespace(
                hidden_size=8,
                num_attention_heads=2,
                n_inner=16,
                vocab_size=32,
            ),
        )
    )

    module.GPT2RefLMHeadModel._init_ref_buffers(subject, vllm_config)

    assert subject._buf_final_logits.dtype is head_dtype


def _call_forwards_keyword(function: object, keyword: str) -> bool:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return any(
        item.arg == keyword
        and isinstance(item.value, ast.Name)
        and item.value.id == keyword
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for item in node.keywords
    )


@pytest.mark.parametrize("module_name", QWEN3_MODULES)
def test_qwen3_variants_forward_per_layer_sliding_window(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)

    for model_type in (module.Qwen3Attention, module.Qwen3DecoderLayer):
        assert "per_layer_sliding_window" in inspect.signature(
            model_type.__init__
        ).parameters
        assert _call_forwards_keyword(
            model_type.__init__, "per_layer_sliding_window"
        )


def test_llama_ref_loader_passes_the_hf_to_vllm_mapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tests.oracles.llama_ref")
    calls: dict[str, object] = {}

    class FakeLoader:
        def __init__(self, model: object):
            calls["model"] = model

        def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
            *,
            mapper: object,
        ) -> set[str]:
            calls["weights"] = list(weights)
            calls["mapper"] = mapper
            return {"loaded"}

    monkeypatch.setattr(module, "AutoWeightsLoader", FakeLoader)
    model = module.LlamaRefForCausalLM.__new__(module.LlamaRefForCausalLM)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(tie_word_embeddings=False),
    )
    weight = torch.zeros(2, 2)

    loaded = model.load_weights(
        [("model.layers.0.self_attn.q_proj.weight", weight)]
    )

    assert loaded == {"loaded"}
    assert calls["model"] is model
    assert calls["mapper"] is model.hf_to_vllm_mapper
    mapped_name, mapped_weight = next(
        iter(
            model.hf_to_vllm_mapper.apply(
                [("model.layers.0.self_attn.q_proj.weight", weight)]
            )
        )
    )
    assert mapped_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert mapped_weight is weight
    assert mapped_weight.shard_id == "q"


def _make_fake_qwen2_moe_block(
    monkeypatch: pytest.MonkeyPatch,
):
    _require_unmodified_router()
    module = importlib.import_module("dmi_vllm_integration.models.qwen2_moe")
    router_module = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.router.fused_moe_router"
    )
    config_module = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.config"
    )

    class FakeRouter(router_module.FusedMoERouter):
        routing_method_type = config_module.RoutingMethodType.Default

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def set_capture_fn(self, capture_fn) -> None:
            pass

        def _select_experts(
            self,
            hidden_states: torch.Tensor,
            *args,
            **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls += 1
            rows = hidden_states.shape[0]
            weights = torch.full(
                (rows, 2), float(self.calls), dtype=torch.float64
            )
            ids = torch.full((rows, 2), self.calls, dtype=torch.int64)
            return weights, ids

    class FakeExperts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = FakeRouter()
            self.used_route = None

        def forward(
            self,
            *,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> torch.Tensor:
            self.used_route = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            return hidden_states

    class FakeGate(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor):
            return hidden_states.new_zeros((hidden_states.shape[0], 4)), None

    def fake_base_init(self, *args, **kwargs) -> None:
        del args, kwargs
        torch.nn.Module.__init__(self)
        self.experts = FakeExperts()
        self.gate = FakeGate()

    monkeypatch.setattr(
        module._Qwen2MoeSparseMoeBlock,
        "__init__",
        fake_base_init,
    )
    return module.Qwen2MoeSparseMoeBlock()


def test_qwen2_moe_captures_the_route_consumed_by_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = _make_fake_qwen2_moe_block(monkeypatch)
    hidden_states = torch.randn(3, 8)
    block._buf_topk_weights = torch.empty(3, 2, dtype=torch.float32)
    block._buf_topk_ids = torch.empty(3, 2, dtype=torch.int32)
    assert torch.equal(block(hidden_states), hidden_states)

    assert block.experts.router.calls == 1
    used_weights, used_ids = block.experts.used_route
    assert torch.equal(block._buf_topk_weights, used_weights.to(torch.float32))
    assert torch.equal(block._buf_topk_ids, used_ids.to(torch.int32))


@pytest.mark.parametrize(
    (
        "local_hook_name",
        "monolithic",
        "internal_router",
        "naive_dispatch",
        "pcp_gather",
        "error",
    ),
    (
        ("topk_ids", True, False, False, False, "requires a modular MoE backend"),
        ("topk_weights", False, False, False, False, None),
        ("topk_ids", False, False, True, False, "dispatch cross-rank token rows"),
        ("topk_ids", False, False, False, True, "dispatch cross-rank token rows"),
        ("router_logits", True, False, True, True, None),
        (
            "router_logits",
            True,
            True,
            False,
            False,
            "requires a modular MoE backend",
        ),
    ),
)
def test_qwen2_moe_validates_selected_routing_capture_backend(
    monkeypatch: pytest.MonkeyPatch,
    local_hook_name: str,
    monolithic: bool,
    internal_router: bool,
    naive_dispatch: bool,
    pcp_gather: bool,
    error: str | None,
) -> None:
    from dmi_vllm_integration.adapter import VLLMAdaptor
    from dmi_vllm_integration.dmi_api import (
        HOOK_TYPE_ROUTER_LOGITS,
        HOOK_TYPE_TOPK_IDS,
        HOOK_TYPE_TOPK_WEIGHTS,
        HookSpec,
    )

    runner_module = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.layer"
    )

    class FakeMoERunner:
        is_monolithic = monolithic
        gate = object() if internal_router else None
        do_naive_dispatch_combine = naive_dispatch
        moe_config = SimpleNamespace(
            pcp_size=2 if pcp_gather else 1,
            moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
        )

    monkeypatch.setattr(runner_module, "MoERunner", FakeMoERunner)
    hook_types = {
        "router_logits": HOOK_TYPE_ROUTER_LOGITS,
        "topk_ids": HOOK_TYPE_TOPK_IDS,
        "topk_weights": HOOK_TYPE_TOPK_WEIGHTS,
    }
    local_spec = HookSpec(hook_types[local_hook_name], None, layer_no=0)
    model = SimpleNamespace(modules=lambda: (FakeMoERunner(),))

    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            VLLMAdaptor._validate_moe_routing_capture(model, [local_spec])
    else:
        VLLMAdaptor._validate_moe_routing_capture(model, [local_spec])


def test_qwen2_moe_compare_keeps_topk_weights_in_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_unmodified_router()
    module = importlib.import_module(
        "tests.oracles.qwen2_moe_compare"
    )
    monkeypatch.setattr(
        module.torch,
        "empty",
        lambda *shape, dtype, **kwargs: SimpleNamespace(dtype=dtype),
    )
    monkeypatch.setattr(
        module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    mlp = SimpleNamespace(hook_router_logits=object())
    layer = SimpleNamespace(self_attn=SimpleNamespace(), mlp=mlp)
    subject = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            intermediate_size=16,
        ),
        model=SimpleNamespace(
            start_layer=0,
            end_layer=1,
            layers=[layer],
        ),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
    )

    module.Qwen2MoeCompareForCausalLM.allocate_compare_buffers(
        subject,
        max_len=4,
        vllm_config=vllm_config,
    )

    assert mlp._buf_router_logits.dtype is torch.bfloat16
    assert mlp._buf_topk_ids.dtype is torch.int32
    assert mlp._buf_topk_weights.dtype is torch.float32
