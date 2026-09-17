# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The Baidu team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

# Adapted from vllm/model_executor/models/ernie45.py in official vLLM 0.29.0.
"""Bounded ERNIE 4.5 dense variant using DMI's hooked Llama model."""

from __future__ import annotations

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig

from dmi_vllm_integration.models.llama import LlamaPForCausalLM
from vllm.model_executor.models.utils import PPMissingLayer


def _apply_ernie45_attention_contract(model) -> None:
    """Replay the exact post-construction adjustments from upstream ERNIE."""

    for layer in model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        layer.self_attn.rotary_emb.is_neox_style = False
        layer.self_attn.o_proj.bias = None
        layer.self_attn.o_proj.skip_bias_add = True


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Ernie4_5PForCausalLM(LlamaPForCausalLM):
    """ERNIE 4.5 dense text model with the upstream attention adjustments."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _apply_ernie45_attention_contract(self.model)


__all__ = ["Ernie4_5PForCausalLM"]
