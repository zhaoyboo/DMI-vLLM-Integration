# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2025 The Swiss AI Initiative.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate the architectural differences made by
# the Swiss AI Initiative that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from vllm/model_executor/models/apertus.py in official vLLM 0.29.0.
"""Apertus dense decoder with DMI observation hooks."""

from __future__ import annotations

from itertools import islice

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors

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
    HookSpec,
)

from vllm.model_executor.models.apertus import (
    ApertusAttention,
    ApertusDecoderLayer,
    ApertusForCausalLM,
    ApertusMLP,
    ApertusModel,
)
from vllm.model_executor.models.utils import PPMissingLayer


def _add_hook_points(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        setattr(module, f"hook_{name}", HookPoint())


class ApertusPMLP(ApertusMLP):
    """Apertus xIELU MLP with its pre-down-projection boundary exposed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hook_post.enabled:
            return super().forward(x)
        x, _ = self.up_proj(x)
        x = self.act_fn(x)
        self.hook_post(x)
        x, _ = self.down_proj(x)
        return x


class ApertusPAttention(ApertusAttention):
    """Apertus attention with post-QK-norm, pre-RoPE Q/K hooks."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not any(
            hook.enabled
            for hook in (self.hook_q, self.hook_k, self.hook_v, self.hook_z)
        ):
            return super().forward(positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q_by_head).view_as(q)
        k = self.k_norm(k_by_head).view_as(k)
        if self.hook_q.enabled:
            self.hook_q(q.view(-1, self.num_heads, self.head_dim))
        if self.hook_k.enabled:
            self.hook_k(k.view(-1, self.num_kv_heads, self.head_dim))
        if self.hook_v.enabled:
            self.hook_v(v.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.hook_z.enabled:
            self.hook_z(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class ApertusPDecoderLayer(ApertusDecoderLayer):
    """Apertus fused pre-norm block with hooks in execution order."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.self_attn.__class__ = ApertusPAttention
        self.mlp.__class__ = ApertusPMLP
        _add_hook_points(
            self,
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
        _add_hook_points(self.self_attn, ("q", "k", "v", "z"))
        _add_hook_points(self.mlp, ("post",))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_hooks = (
            self.hook_resid_pre,
            self.hook_ln1,
            self.hook_attn_out,
            self.hook_resid_mid,
            self.hook_ln2,
            self.hook_mlp_in,
            self.hook_mlp_out,
        )
        if not any(hook.enabled for hook in layer_hooks):
            return super().forward(positions, hidden_states, residual)

        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_layernorm(hidden_states)
        else:
            hidden_states, residual = self.attention_layernorm(
                hidden_states, residual
            )
        if self.hook_resid_pre.enabled:
            # The fused add-RMSNorm returns the completed residual.  Capture
            # that authoritative value instead of inserting a duplicate
            # hidden_states + residual CUDA node ahead of the fused kernel.
            self.hook_resid_pre(residual)
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        if self.hook_attn_out.enabled:
            self.hook_attn_out(hidden_states)

        hidden_states, residual = self.feedforward_layernorm(
            hidden_states, residual
        )
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(residual)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
        return hidden_states, residual


class ApertusPModel(ApertusModel):
    """Concrete compiled Apertus backbone containing model-wide hooks."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = ApertusPDecoderLayer,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=layer_type,
        )
        _add_hook_points(self, ("embed", "resid_final", "final_ln"))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if not any(
            hook.enabled
            for hook in (self.hook_embed, self.hook_resid_final, self.hook_final_ln)
        ):
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )

        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            if self.hook_embed.enabled:
                self.hook_embed(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], 0, hidden_states, residual
        )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, final_residual = self.norm(hidden_states, residual)
        if self.hook_resid_final.enabled:
            self.hook_resid_final(final_residual)
        if self.hook_final_ln.enabled:
            self.hook_final_ln(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class ApertusPForCausalLM(ApertusForCausalLM):
    """Apertus causal LM with a truthful DMI hook manifest."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model_type: type[ApertusPModel] = ApertusPModel,
        layer_type: type[nn.Module] = ApertusPDecoderLayer,
    ) -> None:
        self._dmi_model_type = model_type
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=layer_type,
        )
        del self._dmi_model_type
        self.hook_token_ids = HookPoint()
        self.hook_final_logits = HookPoint()

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = ApertusPDecoderLayer,
    ) -> ApertusPModel:
        return self._dmi_model_type(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=layer_type,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if (
            input_ids is not None
            and get_pp_group().is_first_rank
            and self.hook_token_ids.enabled
        ):
            self.hook_token_ids(input_ids)
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is not None and self.hook_final_logits.enabled:
            self.hook_final_logits(logits)
        return logits

    @staticmethod
    def _layer_hook_specs(
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        attention = None if layer is None else layer.self_attn
        mlp = None if layer is None else layer.mlp

        def hook(module, name: str):
            return None if module is None else getattr(module, f"hook_{name}")

        def spec(hook_type: int, module, name: str) -> HookSpec:
            return HookSpec(
                hook_type,
                hook(module, name),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
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
            spec(HOOK_TYPE_MLP_POST, mlp, "post"),
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
        layer_numbers = (
            range(self.config.num_hidden_layers)
            if model_wide
            else range(model.start_layer, model.end_layer)
        )
        for layer_no in layer_numbers:
            layer = None
            if not model_wide:
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
    "ApertusPAttention",
    "ApertusPDecoderLayer",
    "ApertusPForCausalLM",
    "ApertusPMLP",
    "ApertusPModel",
]
