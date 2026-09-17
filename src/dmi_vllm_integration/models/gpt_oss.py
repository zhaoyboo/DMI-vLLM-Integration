# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from vllm/model_executor/models/gpt_oss.py in official vLLM 0.29.0.
"""GPT-OSS decoder with DMI observation hooks."""

from __future__ import annotations

import torch
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
    HookSpec,
)
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.utils import rocm_unquantized_gemm
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.gpt_oss import (
    GptOssForCausalLM,
    GptOssModel,
    MLPBlock,
    OAIAttention,
    TransformerBlock,
)
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

from dmi_vllm_integration.patches import (
    apply_fused_moe_router_observer_patch,
)


apply_fused_moe_router_observer_patch()


def _add_hook_points(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        setattr(module, f"hook_{name}", HookPoint())


def _capture_compare_buffer(
    module: nn.Module,
    name: str,
    value: torch.Tensor,
) -> None:
    buffer = getattr(module, f"_buf_{name}", None)
    if buffer is not None:
        buffer[: value.shape[0]].copy_(value)


class GptOssPAttention(OAIAttention):
    """GPT-OSS attention with pre-RoPE Q/K/V and pre-o-proj Z hooks."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hooks = (self.hook_q, self.hook_k, self.hook_v, self.hook_z)
        if not any(hook.enabled for hook in hooks):
            return super().forward(hidden_states, positions)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.hook_q.enabled:
            q_by_head = q.unflatten(
                -1,
                (self.num_local_attention_heads, self.head_dim),
            )
            self.hook_q(q_by_head)
            _capture_compare_buffer(self, "q", q_by_head)
        if self.hook_k.enabled:
            k_by_head = k.unflatten(
                -1,
                (self.num_local_key_value_heads, self.head_dim),
            )
            self.hook_k(k_by_head)
            _capture_compare_buffer(self, "k", k_by_head)
        if self.hook_v.enabled:
            v_by_head = v.unflatten(
                -1,
                (self.num_local_key_value_heads, self.head_dim),
            )
            self.hook_v(v_by_head)
            _capture_compare_buffer(self, "v", v_by_head)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.hook_z.enabled:
            self.hook_z(attn_output)
            _capture_compare_buffer(self, "z", attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class GptOssPMLP(MLPBlock):
    """GPT-OSS experts with token-major routing observation."""

    def _observe_routing(
        self,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        del router_logits
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(torch.float32)
        if self.hook_topk_ids.enabled:
            self.hook_topk_ids(topk_ids)
            _capture_compare_buffer(self, "topk_ids", topk_ids)
        if self.hook_topk_weights.enabled:
            self.hook_topk_weights(topk_weights)
            _capture_compare_buffer(self, "topk_weights", topk_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing_hooks = (
            self.hook_router_logits,
            self.hook_topk_ids,
            self.hook_topk_weights,
        )
        if not any(hook.enabled for hook in routing_hooks):
            return super().forward(x)

        num_tokens = x.shape[0]
        if self.is_sequence_parallel:
            x = sequence_parallel_chunk(x)

        if current_platform.is_rocm():
            router_logits = rocm_unquantized_gemm(
                self,
                x[:, : self.hidden_size],
                self.router.weight,
                self.router.bias,
            )
        else:
            router_logits = self.router(x)
        if self.hook_router_logits.enabled:
            self.hook_router_logits(router_logits)
            _capture_compare_buffer(self, "router_logits", router_logits)
        x = self.experts(hidden_states=x, router_logits=router_logits)[
            :, : self.hidden_size
        ]
        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x.contiguous(), 0)
            x = x[:num_tokens]
        return x


class GptOssPTransformerBlock(TransformerBlock):
    """One GPT-OSS block with fused-residual boundaries exposed."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direct_hooks = (
            self.hook_resid_pre,
            self.hook_ln1,
            self.hook_attn_out,
            self.hook_resid_mid,
            self.hook_ln2,
            self.hook_mlp_in,
            self.hook_mlp_out,
        )
        if not any(hook.enabled for hook in direct_hooks):
            return super().forward(hidden_states, positions, residual)

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.hook_resid_pre.enabled:
            self.hook_resid_pre(residual)
            _capture_compare_buffer(self, "resid_pre", residual)
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)
            _capture_compare_buffer(self, "ln1", hidden_states)

        hidden_states = self.attn(hidden_states, positions)
        if self.hook_attn_out.enabled:
            self.hook_attn_out(hidden_states)
            _capture_compare_buffer(self, "attn_out", hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(residual)
            _capture_compare_buffer(self, "resid_mid", residual)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
            _capture_compare_buffer(self, "ln2", hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
            _capture_compare_buffer(self, "mlp_in", hidden_states)
        output = self.mlp(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(output)
            _capture_compare_buffer(self, "mlp_out", output)
        return output, residual


def _instrument_gpt_oss_model(model: GptOssModel) -> GptOssPModel:
    """Attach hooks after the DMI backbone compile wrapper is initialized."""

    if not isinstance(model, GptOssPModel):
        raise TypeError("GPT-OSS backbone must be constructed as its DMI class")
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.__class__ = GptOssPTransformerBlock
        layer.attn.__class__ = GptOssPAttention
        layer.mlp.__class__ = GptOssPMLP
        _add_hook_points(
            layer,
            (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ),
        )
        _add_hook_points(layer.attn, ("q", "k", "v", "z"))
        _add_hook_points(
            layer.mlp,
            ("router_logits", "topk_ids", "topk_weights"),
        )
        layer.mlp.experts.router.set_routing_observer(layer.mlp._observe_routing)
    return model


class GptOssPModel(GptOssModel):
    """Concrete GPT-OSS backbone with embedding and final hooks."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _instrument_gpt_oss_model(self)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        final_hooks = (self.hook_embed, self.hook_resid_final, self.hook_final_ln)
        if not any(hook.enabled for hook in final_hooks):
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )

        if get_pp_group().is_first_rank:
            x = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
            if self.hook_embed.enabled:
                self.hook_embed(x)
                _capture_compare_buffer(self, "embed", x)
        else:
            assert intermediate_tensors is not None
            x = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [],
            self.start_layer,
            x,
            residual,
        )
        for layer_no in range(self.start_layer, self.end_layer):
            x, residual = self.layers[layer_no](x, positions, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states,
                layer_no + 1,
                x,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": x, "residual": residual})
        x, final_residual = self.norm(x, residual)
        if self.hook_resid_final.enabled:
            self.hook_resid_final(final_residual)
            _capture_compare_buffer(self, "resid_final", final_residual)
        if self.hook_final_ln.enabled:
            self.hook_final_ln(x)
            _capture_compare_buffer(self, "final_ln", x)
        if aux_hidden_states:
            return x, aux_hidden_states
        return x


class GptOssPForCausalLM(GptOssForCausalLM):
    """GPT-OSS causal LM with a truthful DMI hook manifest."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        model_type: type[GptOssPModel] = GptOssPModel,
    ) -> None:
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model = model_type(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.hook_token_ids = HookPoint()
        self.hook_final_logits = HookPoint()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if (
            input_ids is not None
            and get_pp_group().is_first_rank
            and self.hook_token_ids.enabled
        ):
            self.hook_token_ids(input_ids)
            _capture_compare_buffer(self, "token_ids", input_ids)
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = super().compute_logits(hidden_states)
        if self.hook_final_logits.enabled:
            self.hook_final_logits(logits)
            _capture_compare_buffer(self, "final_logits", logits)
        return logits

    @staticmethod
    def _layer_hook_specs(
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        attention = None if layer is None else layer.attn
        mlp = None if layer is None else layer.mlp

        def hook(module, name: str):
            return None if module is None else getattr(module, f"hook_{name}")

        def spec(hook_type: int, module, name: str, **kwargs) -> HookSpec:
            return HookSpec(
                hook_type,
                hook(module, name),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
                **kwargs,
            )

        return [
            spec(HOOK_TYPE_RESID_PRE, layer, "resid_pre"),
            spec(HOOK_TYPE_LN1, layer, "ln1"),
            spec(HOOK_TYPE_Q, attention, "q"),
            spec(HOOK_TYPE_K, attention, "k"),
            spec(HOOK_TYPE_V, attention, "v"),
            spec(HOOK_TYPE_Z, attention, "z"),
            spec(HOOK_TYPE_ATTN_OUT, layer, "attn_out"),
            spec(HOOK_TYPE_RESID_MID, layer, "resid_mid"),
            spec(HOOK_TYPE_LN2, layer, "ln2"),
            spec(HOOK_TYPE_MLP_IN, layer, "mlp_in"),
            spec(HOOK_TYPE_ROUTER_LOGITS, mlp, "router_logits"),
            spec(HOOK_TYPE_TOPK_IDS, mlp, "topk_ids", dtype=torch.int32),
            spec(
                HOOK_TYPE_TOPK_WEIGHTS,
                mlp,
                "topk_weights",
                dtype=torch.float32,
            ),
            spec(HOOK_TYPE_MLP_OUT, layer, "mlp_out"),
        ]

    def get_hook_specs(self, model_wide: bool = False) -> list[HookSpec]:
        model = self.model
        specs = [
            HookSpec(
                HOOK_TYPE_TOKEN_IDS,
                None if model_wide else self.hook_token_ids,
                dtype=torch.int32,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_EMBED,
                None if model_wide else model.hook_embed,
                dim0_is_actual_tokens=True,
            ),
        ]
        for layer_no in range(self.config.num_hidden_layers):
            layer = None
            if not model_wide and model.start_layer <= layer_no < model.end_layer:
                candidate = model.layers[layer_no]
                if not isinstance(candidate, PPMissingLayer):
                    layer = candidate
            specs.extend(self._layer_hook_specs(layer_no, layer))
        specs.extend(
            [
                HookSpec(
                    HOOK_TYPE_RESID_FINAL,
                    None if model_wide else model.hook_resid_final,
                    dim0_is_actual_tokens=True,
                ),
                HookSpec(
                    HOOK_TYPE_FINAL_LN,
                    None if model_wide else model.hook_final_ln,
                    dim0_is_actual_tokens=True,
                ),
                HookSpec(
                    HOOK_TYPE_FINAL_LOGITS,
                    None if model_wide else self.hook_final_logits,
                ),
            ]
        )
        return specs


__all__ = [
    "GptOssPAttention",
    "GptOssPForCausalLM",
    "GptOssPMLP",
    "GptOssPModel",
    "GptOssPTransformerBlock",
    "_instrument_gpt_oss_model",
]
