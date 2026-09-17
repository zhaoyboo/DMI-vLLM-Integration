# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The vLLM team.
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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

# Adapted from vllm/model_executor/models/gemma4.py in official vLLM 0.29.0.
"""Gemma 4 E2B decoder-boundary monitoring behind the public MM wrapper.

The pinned E2B checkpoint has heterogeneous attention head dimensions and
heterogeneous MLP widths.  DMI's current shape contract describes one global
head/intermediate dimension, so this variant intentionally exposes only the
decoder boundaries whose shapes are uniform and exact.
"""

from __future__ import annotations

from itertools import islice
from typing import Any

import torch
from dmi_vllm_integration.dmi_api import HookPoint
from dmi_vllm_integration.dmi_api import (
    HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_EMBED,
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_LN1,
    HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_TOKEN_IDS,
    HookSpec,
)
from torch import nn
from transformers import AutoModel

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.gemma4 import (
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
    Gemma4MoE,
    Gemma4Model,
    _get_text_config,
)
from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
)
from vllm.model_executor.models.transformers.utils import recursive_replace_linear
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix


def _require_supported_gemma4_e2b_config(
    config: Any,
    parallel_config: Any,
    quant_config: Any = None,
    dtype: torch.dtype | None = None,
    *,
    speculative_config: Any = None,
    kv_sharing_fast_prefill: bool = False,
) -> None:
    """Require only Gemma 4 branches implemented by DMI's hooked forward."""

    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise NotImplementedError("DMI Gemma 4 support requires a nested text config")
    if getattr(text_config, "enable_moe_block", False):
        raise NotImplementedError(
            "DMI Gemma 4 support currently requires enable_moe_block=False "
            "because its hooked decoder instruments the dense MLP branch"
        )
    if kv_sharing_fast_prefill:
        raise NotImplementedError(
            "DMI Gemma 4 support currently excludes KV-sharing fast prefill "
            "because its hooked decoder does not implement that upstream branch"
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


class Gemma4PDecoderLayer(Gemma4DecoderLayer):
    """Gemma 4 layer exposing only uniform hidden-size boundaries."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        per_layer_input: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
                positions,
                hidden_states,
                residual,
                per_layer_input=per_layer_input,
                **kwargs,
            )

        residual = hidden_states
        self.hook_resid_pre(residual)
        _capture_compare_buffer(self, "resid_pre", residual)

        hidden_states = self.input_layernorm(residual)
        self.hook_ln1(hidden_states)
        _capture_compare_buffer(self, "ln1", hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        self.hook_attn_out(hidden_states)
        _capture_compare_buffer(self, "attn_out", hidden_states)

        hidden_states = hidden_states + residual
        residual = hidden_states
        self.hook_resid_mid(residual)
        _capture_compare_buffer(self, "resid_mid", residual)

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        self.hook_ln2(hidden_states)
        _capture_compare_buffer(self, "ln2", hidden_states)
        self.hook_mlp_in(hidden_states)
        _capture_compare_buffer(self, "mlp_in", hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        self.hook_mlp_out(hidden_states)
        _capture_compare_buffer(self, "mlp_out", hidden_states)
        hidden_states = hidden_states + residual

        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            contribution = self.per_layer_projection(gate * per_layer_input)
            contribution = self.post_per_layer_input_norm(contribution)
            hidden_states = hidden_states + contribution

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


class Gemma4PModel(Gemma4Model):
    """Gemma 4 decoder backbone with embedding/final hidden boundaries."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        hooks = (self.hook_embed, self.hook_resid_final, self.hook_final_ln)
        if not any(hook.enabled for hook in hooks):
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                per_layer_inputs,
                **kwargs,
            )

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_inputs
                )
            else:
                hidden_states = self.embed_input_ids(input_ids)
                per_layer_embeds = self.get_per_layer_inputs(input_ids)
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_embeds
                )
            self.hook_embed(hidden_states)
            _capture_compare_buffer(self, "embed", hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            if per_layer_inputs is not None:
                per_layer_inputs = intermediate_tensors["per_layer_inputs"]

        residual = None
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            layer_per_input = None
            if per_layer_inputs is not None:
                actual_layer_idx = self.start_layer + layer_idx
                layer_per_input = per_layer_inputs[:, actual_layer_idx, :]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                per_layer_input=layer_per_input,
                **kwargs,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states,
                layer_idx + 1,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            tensors = {"hidden_states": hidden_states}
            if per_layer_inputs is not None:
                tensors["per_layer_inputs"] = per_layer_inputs
            return IntermediateTensors(tensors)

        self.hook_resid_final(hidden_states)
        _capture_compare_buffer(self, "resid_final", hidden_states)
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        self.hook_final_ln(hidden_states)
        _capture_compare_buffer(self, "final_ln", hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Gemma4PForCausalLM(Gemma4ForCausalLM):
    """Instrumented native language model owned by the MM wrapper."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model_type: type[Gemma4PModel] = Gemma4PModel,
    ) -> None:
        config = _get_text_config(vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config

        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.model = model_type(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.moe_layers: list[nn.Module] = []
        example_moe: Gemma4MoE | None = None
        for layer in self.model.layers:
            if hasattr(layer, "moe") and isinstance(layer.moe, Gemma4MoE):
                example_moe = layer.moe
                self.moe_layers.append(layer.moe.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example_moe is not None:
            self.num_logical_experts = example_moe.num_experts
            self.num_physical_experts = example_moe.num_experts
            self.num_local_physical_experts = example_moe.num_experts
            self.num_routed_experts = example_moe.num_experts
        else:
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0

        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_redundant_experts = 0
        _instrument_gemma4_language_model(self)

    def _layer_hook_specs(
        self,
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        def hook(name: str):
            return None if layer is None else getattr(layer, f"hook_{name}")

        def spec(hook_type: int, name: str) -> HookSpec:
            return HookSpec(
                hook_type,
                hook(name),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            )

        return [
            spec(HOOK_TYPE_RESID_PRE, "resid_pre"),
            spec(HOOK_TYPE_LN1, "ln1"),
            spec(HOOK_TYPE_ATTN_OUT, "attn_out"),
            spec(HOOK_TYPE_RESID_MID, "resid_mid"),
            spec(HOOK_TYPE_LN2, "ln2"),
            spec(HOOK_TYPE_MLP_IN, "mlp_in"),
            spec(HOOK_TYPE_MLP_OUT, "mlp_out"),
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


def _instrument_gemma4_language_model(
    language_model: Gemma4ForCausalLM,
) -> Gemma4PForCausalLM:
    """Attach child hooks after constructing the concrete DMI backbone."""

    if not isinstance(language_model, Gemma4PForCausalLM):
        raise TypeError("Gemma 4 language model must be constructed as its DMI class")
    _add_hook_points(language_model, ("token_ids", "final_logits"))
    model = language_model.model
    if not isinstance(model, Gemma4PModel):
        raise TypeError("Gemma 4 backbone must be constructed as its DMI class")
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.__class__ = Gemma4PDecoderLayer
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
    return language_model


class Gemma4PForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Public Gemma 4 multimodal model exporting decoder boundaries only."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        language_model_type: type[Gemma4PForCausalLM] = Gemma4PForCausalLM,
    ) -> None:
        _require_supported_gemma4_e2b_config(
            vllm_config.model_config.hf_config,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
            speculative_config=vllm_config.speculative_config,
            kv_sharing_fast_prefill=(vllm_config.cache_config.kv_sharing_fast_prefill),
        )

        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.model_dtype = vllm_config.model_config.dtype
        self.vllm_config = vllm_config

        if quant_config and quant_config.get_name() in [
            "bitsandbytes",
            "torchao",
            "compressed-tensors",
        ]:
            tower_quant = quant_config
        else:
            vision_cfg = config.vision_config
            quantizable = (
                vision_cfg.hidden_size % 64 == 0
                and vision_cfg.intermediate_size % 64 == 0
            )
            tower_quant = quant_config if quantizable else None

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = AutoModel.from_config(config=config.vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(
                config.vision_config,
                config.text_config,
                quant_config=tower_quant,
                prefix=maybe_prefix(prefix, "embed_vision"),
            )
            recursive_replace_linear(
                self.vision_tower,
                tower_quant,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

        if config.audio_config is not None:
            with self._mark_tower_model(vllm_config, "audio"):
                self.audio_tower = AutoModel.from_config(config=config.audio_config)
                self.audio_tower.post_init()
                self.embed_audio = Gemma4MultimodalEmbedder(
                    config.audio_config,
                    config.text_config,
                    quant_config=tower_quant,
                    prefix=maybe_prefix(prefix, "embed_audio"),
                )
                recursive_replace_linear(
                    self.audio_tower,
                    tower_quant,
                    prefix=maybe_prefix(prefix, "audio_tower"),
                )
        else:
            self.audio_tower = None
            self.embed_audio = None

        with self._mark_language_model(vllm_config):
            inner_vllm_config = vllm_config.with_hf_config(
                config.text_config,
                architectures=["Gemma4ForCausalLM"],
            )
            self.language_model = initialize_model(
                vllm_config=inner_vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
                model_class=language_model_type,
            )

            ple_dim = config.text_config.hidden_size_per_layer_input
            if ple_dim is not None and ple_dim > 0:
                embed = self.language_model.model.embed_tokens
                self.per_layer_embeddings = torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.num_hidden_layers,
                    ple_dim,
                    device=next(embed.parameters()).device,
                    dtype=vllm_config.model_config.dtype,
                )
            else:
                self.per_layer_embeddings = None

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self._full_attn_layer_idxs: frozenset[int] = frozenset()
        text_config = config.text_config
        if getattr(text_config, "use_bidirectional_attention", None) == "vision":
            layer_types = getattr(text_config, "layer_types", None)
            if layer_types:
                self._full_attn_layer_idxs = frozenset(
                    i
                    for i, layer_type in enumerate(layer_types)
                    if layer_type != "sliding_attention"
                )

        self.moe_layers = self.language_model.moe_layers
        self.num_moe_layers = self.language_model.num_moe_layers
        self.num_logical_experts = self.language_model.num_logical_experts
        self.num_physical_experts = self.language_model.num_physical_experts
        self.num_local_physical_experts = (
            self.language_model.num_local_physical_experts
        )
        self.num_routed_experts = self.language_model.num_routed_experts
        self.num_expert_groups = self.language_model.num_expert_groups
        self.num_shared_experts = self.language_model.num_shared_experts
        self.num_redundant_experts = self.language_model.num_redundant_experts

        gen_cfg = vllm_config.model_config.try_get_generation_config()
        self._suppress_token_ids = (
            gen_cfg.get("suppress_tokens") if gen_cfg else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if self.language_model.hook_token_ids.enabled:
            self.language_model.hook_token_ids(input_ids)
            _capture_compare_buffer(self.language_model, "token_ids", input_ids)
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is not None and self.language_model.hook_final_logits.enabled:
            self.language_model.hook_final_logits(logits)
            _capture_compare_buffer(self.language_model, "final_logits", logits)
        return logits

    def get_hook_specs(self, model_wide: bool = False) -> list[HookSpec]:
        return self.language_model.get_hook_specs(model_wide=model_wide)


__all__ = [
    "Gemma4PDecoderLayer",
    "Gemma4PForCausalLM",
    "Gemma4PForConditionalGeneration",
    "Gemma4PModel",
    "_instrument_gemma4_language_model",
    "_require_supported_gemma4_e2b_config",
]
