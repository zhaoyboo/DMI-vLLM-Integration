# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
# All rights reserved.
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

# Adapted from vllm/model_executor/models/mllama4.py in official vLLM 0.29.0.
"""Llama 4 multimodal wrapper exporting only language-decoder DMI hooks."""

from __future__ import annotations

from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.model_executor.models.mllama4 import (
    Llama4ForConditionalGeneration,
    Llama4MultiModalProjector,
    Llama4VisionModel,
)
from vllm.model_executor.models.utils import maybe_prefix

from dmi_vllm_integration.models.llama4 import (
    Llama4PForCausalLM,
    _require_supported_llama4_scout_config,
)


class Llama4PForConditionalGeneration(Llama4ForConditionalGeneration):
    """Preserve Llama 4 public multimodal behavior and monitor its decoder."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        language_model_type: type[Llama4PForCausalLM] = Llama4PForCausalLM,
    ) -> None:
        _require_supported_llama4_scout_config(
            vllm_config.model_config.hf_config,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
        )

        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config

        with self._mark_tower_model(vllm_config, "image"):
            with set_current_vllm_config(vllm_config):
                self.vision_model = Llama4VisionModel(
                    config=config.vision_config,
                    quant_config=None,
                    prefix=maybe_prefix(prefix, "vision_model"),
                )

            self.multi_modal_projector = Llama4MultiModalProjector(
                config=self.config,
                quant_config=None,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = initialize_model(
                vllm_config=vllm_config.with_hf_config(
                    config.text_config,
                    ["LlamaForCausalLM"],
                ),
                prefix=maybe_prefix(prefix, "language_model"),
                model_class=language_model_type,
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self.num_expert_groups = 1
        self.num_logical_experts = self.language_model.num_logical_experts
        self.num_physical_experts = self.language_model.num_physical_experts
        self.num_local_physical_experts = self.language_model.num_local_physical_experts
        self.num_routed_experts = self.language_model.num_routed_experts
        self.num_shared_experts = self.language_model.num_shared_experts
        self.num_redundant_experts = self.language_model.num_redundant_experts
        self.moe_layers = self.language_model.moe_layers
        self.num_moe_layers = len(self.moe_layers)

    def get_hook_specs(self, model_wide: bool = False):
        return self.language_model.get_hook_specs(model_wide=model_wide)


__all__ = ["Llama4PForConditionalGeneration"]
