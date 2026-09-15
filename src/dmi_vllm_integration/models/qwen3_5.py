# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

# Adapted from vllm/model_executor/models/qwen3_5.py in official vLLM 0.27.1.
"""Qwen3.6 dense multimodal model with decoder-only DMI hooks."""

from __future__ import annotations

from itertools import islice

import torch
import torch.nn.functional as F
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
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config

from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix


def _require_supported_qwen36_text_config(
    config,
    parallel_config=None,
    quant_config=None,
    dtype=None,
) -> None:
    """Require the dense branches implemented by DMI's Qwen3.6 hooks."""

    if getattr(config, "model_type", None) != "qwen3_5_text":
        raise NotImplementedError(
            "DMI Qwen3.6 support currently requires the dense text variant"
        )
    if getattr(config, "layer_scale", False):
        raise NotImplementedError(
            "DMI Qwen3.6 support currently excludes layer scaling because its "
            "hooked decoder forward does not apply the upstream scale parameters"
        )


def _require_supported_qwen36_config(
    config,
    parallel_config=None,
    quant_config=None,
    dtype=None,
) -> None:
    """Validate only DMI-specific branches of the monitored decoder tier."""

    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise NotImplementedError("DMI Qwen3.6 support requires a nested text config")
    _require_supported_qwen36_text_config(
        text_config,
        parallel_config,
        quant_config,
        dtype,
    )


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


class Qwen3_5PMLP(Qwen3NextMLP):
    """Dense Qwen3.6 MLP exposing post-activation values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hook_post.enabled:
            return super().forward(x)

        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        if self.hook_post.enabled:
            self.hook_post(out)
            _capture_compare_buffer(self, "mlp_post", out)
        out, _ = self.down_proj(out)
        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out
        return out


class Qwen3_5PAttention(Qwen3NextAttention):
    """Full attention with post-MRoPE Q/K/V and gated-Z observations."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hooks = (self.hook_q, self.hook_k, self.hook_v, self.hook_z)
        if not any(hook.enabled for hook in hooks):
            return super().forward(positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        if self.hook_q.enabled:
            q_by_head = q.unflatten(-1, (self.num_heads, self.head_dim))
            self.hook_q(q_by_head)
            _capture_compare_buffer(self, "q", q_by_head)
        if self.hook_k.enabled:
            k_by_head = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            self.hook_k(k_by_head)
            _capture_compare_buffer(self, "k", k_by_head)
        if self.hook_v.enabled:
            v_by_head = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            self.hook_v(v_by_head)
            _capture_compare_buffer(self, "v", v_by_head)
        attn_output = self.attn(q, k, v)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        if self.hook_z.enabled:
            self.hook_z(attn_output)
            _capture_compare_buffer(self, "z", attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3_5PDecoderLayer(Qwen3_5DecoderLayer):
    """Hybrid decoder layer exposing common boundaries only for GDN layers."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hooks = (
            self.hook_resid_pre,
            self.hook_ln1,
            self.hook_attn_out,
            self.hook_resid_mid,
            self.hook_ln2,
            self.hook_mlp_in,
            self.hook_mlp_out,
        )
        if not any(hook.enabled for hook in hooks):
            return super().forward(
                hidden_states,
                residual,
                positions=positions,
                **kwargs,
            )

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

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        if self.hook_attn_out.enabled:
            self.hook_attn_out(hidden_states)
            _capture_compare_buffer(self, "attn_out", hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(residual)
            _capture_compare_buffer(self, "resid_mid", residual)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
            _capture_compare_buffer(self, "ln2", hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
            _capture_compare_buffer(self, "mlp_in", hidden_states)

        hidden_states = self.mlp(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
            _capture_compare_buffer(self, "mlp_out", hidden_states)
        return hidden_states, residual


def _instrument_qwen36_language_model(
    language_model: Qwen3_5ForCausalLM,
    parallel_config=None,
    quant_config=None,
    dtype=None,
) -> Qwen3_5PForCausalLM:
    """Attach hooks after the DMI backbone compile wrapper is initialized."""

    _require_supported_qwen36_text_config(
        language_model.config,
        parallel_config,
        quant_config,
        dtype,
    )
    if not isinstance(language_model, Qwen3_5PForCausalLM):
        raise TypeError("Qwen3.6 language model must be constructed as its DMI class")
    _add_hook_points(language_model, ("token_ids", "final_logits"))
    model = language_model.model
    if not isinstance(model, Qwen3_5PModel):
        raise TypeError("Qwen3.6 backbone must be constructed as its DMI class")
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.__class__ = Qwen3_5PDecoderLayer
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
        layer.mlp.__class__ = Qwen3_5PMLP
        _add_hook_points(layer.mlp, ("post",))
        if layer.layer_type == "full_attention":
            layer.self_attn.__class__ = Qwen3_5PAttention
            _add_hook_points(layer.self_attn, ("q", "k", "v", "z"))
    return language_model


class Qwen3_5PModel(Qwen3_5Model):
    """Concrete Qwen3.6 hybrid backbone with embedding and final hooks."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        hooks = (self.hook_embed, self.hook_resid_final, self.hook_final_ln)
        if not any(hook.enabled for hook in hooks):
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
            residual = None
            if self.hook_embed.enabled:
                self.hook_embed(hidden_states)
                _capture_compare_buffer(self, "embed", hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_no, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states,
                layer_no + 1,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, final_residual = self.norm(hidden_states, residual)
        if self.hook_resid_final.enabled:
            self.hook_resid_final(final_residual)
            _capture_compare_buffer(self, "resid_final", final_residual)
        if self.hook_final_ln.enabled:
            self.hook_final_ln(hidden_states)
            _capture_compare_buffer(self, "final_ln", hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Qwen3_5PForCausalLM(Qwen3_5ForCausalLM):
    """Qwen3.6 language model with a heterogeneous DMI hook manifest."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model_type: type[Qwen3_5PModel] = Qwen3_5PModel,
    ) -> None:
        _require_supported_qwen36_text_config(
            vllm_config.model_config.hf_text_config,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
        )

        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        nn.Module.__init__(self)
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = model_type(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        _instrument_qwen36_language_model(
            self,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
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
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is not None and self.hook_final_logits.enabled:
            self.hook_final_logits(logits)
            _capture_compare_buffer(self, "final_logits", logits)
        return logits

    def _layer_hook_specs(
        self,
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        attention = (
            None
            if layer is None or self.config.layer_types[layer_no] != "full_attention"
            else layer.self_attn
        )
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

        specs = [
            spec(HOOK_TYPE_RESID_PRE, layer, "resid_pre"),
            spec(HOOK_TYPE_LN1, layer, "ln1"),
        ]
        if (
            attention is not None
            or self.config.layer_types[layer_no] == "full_attention"
        ):
            specs.extend(
                [
                    spec(HOOK_TYPE_Q, attention, "q"),
                    spec(HOOK_TYPE_K, attention, "k"),
                    spec(HOOK_TYPE_V, attention, "v"),
                    spec(HOOK_TYPE_Z, attention, "z"),
                ]
            )
        specs.extend(
            [
                spec(HOOK_TYPE_ATTN_OUT, layer, "attn_out"),
                spec(HOOK_TYPE_RESID_MID, layer, "resid_mid"),
                spec(HOOK_TYPE_LN2, layer, "ln2"),
                spec(HOOK_TYPE_MLP_IN, layer, "mlp_in"),
                spec(HOOK_TYPE_MLP_POST, mlp, "post"),
                spec(HOOK_TYPE_MLP_OUT, layer, "mlp_out"),
            ]
        )
        return specs

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


class Qwen3_5PForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Preserve public image/video behavior and monitor only the decoder."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "model",
        language_model_type: type[Qwen3_5PForCausalLM] = Qwen3_5PForCausalLM,
    ) -> None:
        _require_supported_qwen36_config(
            vllm_config.model_config.hf_config,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
        )

        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = language_model_type(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        if (
            input_ids is not None
            and get_pp_group().is_first_rank
            and self.language_model.hook_token_ids.enabled
        ):
            self.language_model.hook_token_ids(input_ids)
            _capture_compare_buffer(self.language_model, "token_ids", input_ids)
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def get_hook_specs(self, model_wide: bool = False):
        return self.language_model.get_hook_specs(model_wide=model_wide)


__all__ = [
    "Qwen3_5PAttention",
    "Qwen3_5PDecoderLayer",
    "Qwen3_5PForCausalLM",
    "Qwen3_5PForConditionalGeneration",
    "Qwen3_5PMLP",
    "Qwen3_5PModel",
    "_instrument_qwen36_language_model",
    "_require_supported_qwen36_config",
    "_require_supported_qwen36_text_config",
]
