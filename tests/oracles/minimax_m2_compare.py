# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The MiniMax AI team.
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

# Adapted from vllm/model_executor/models/minimax_m2.py in official vLLM 0.29.0.
"""MiniMax-M2.7 same-graph compare buffers for DMI storage tests."""

from __future__ import annotations

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size

from dmi_vllm_integration.models.minimax_m2 import MiniMaxM2PForCausalLM
from vllm.model_executor.models.utils import PPMissingLayer


class MiniMaxM2CompareForCausalLM(MiniMaxM2PForCausalLM):
    """Test-only MiniMax model retaining D2D decoder references."""

    def allocate_compare_buffers(
        self,
        max_len: int,
        vllm_config: VllmConfig,
    ) -> None:
        config = self.config
        dtype = vllm_config.model_config.dtype
        device = "cuda"
        tp_size = get_tensor_model_parallel_world_size()
        num_heads = config.num_attention_heads // tp_size
        num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        hidden_size = config.hidden_size
        head_dim = config.head_dim
        model = self.model

        for name in ("embed", "resid_final", "final_ln"):
            setattr(
                model,
                f"_buf_{name}",
                torch.empty(max_len, hidden_size, device=device, dtype=dtype),
            )
        for layer_no in range(model.start_layer, model.end_layer):
            layer = model.layers[layer_no]
            if isinstance(layer, PPMissingLayer):
                continue
            for name in (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ):
                setattr(
                    layer,
                    f"_buf_{name}",
                    torch.empty(max_len, hidden_size, device=device, dtype=dtype),
                )
            attention = layer.self_attn
            attention._buf_q = torch.empty(
                max_len,
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            for name in ("k", "v"):
                setattr(
                    attention,
                    f"_buf_{name}",
                    torch.empty(
                        max_len,
                        num_kv_heads,
                        head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                )
            attention._buf_z = torch.empty(
                max_len,
                num_heads * head_dim,
                device=device,
                dtype=dtype,
            )
            moe = layer.block_sparse_moe
            moe._buf_router_logits = torch.empty(
                max_len,
                config.num_local_experts,
                device=device,
                dtype=torch.float32,
            )
            moe._buf_topk_ids = torch.empty(
                max_len,
                config.num_experts_per_tok,
                device=device,
                dtype=torch.int32,
            )
            moe._buf_topk_weights = torch.empty(
                max_len,
                config.num_experts_per_tok,
                device=device,
                dtype=torch.float32,
            )

        max_requests = vllm_config.scheduler_config.max_num_seqs
        self._buf_final_logits = torch.empty(
            max_requests,
            config.vocab_size,
            device=device,
            dtype=dtype,
        )
        self._buf_token_ids = torch.empty(
            max_len,
            device=device,
            dtype=torch.int32,
        )

    def get_ref_buffers(self) -> dict[str, torch.Tensor]:
        buffers: dict[str, torch.Tensor] = {}
        model = self.model
        for name in ("embed", "resid_final", "final_ln"):
            buffers[name] = getattr(model, f"_buf_{name}")
        for layer_no in range(model.start_layer, model.end_layer):
            layer = model.layers[layer_no]
            if isinstance(layer, PPMissingLayer):
                continue
            for name in (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ):
                buffers[f"{name}_L{layer_no}"] = getattr(
                    layer,
                    f"_buf_{name}",
                )
            for name in ("q", "k", "v", "z"):
                buffers[f"{name}_L{layer_no}"] = getattr(
                    layer.self_attn,
                    f"_buf_{name}",
                )
            for name in ("router_logits", "topk_ids", "topk_weights"):
                buffers[f"{name}_L{layer_no}"] = getattr(
                    layer.block_sparse_moe,
                    f"_buf_{name}",
                )
        buffers["final_logits"] = self._buf_final_logits
        buffers["token_ids"] = self._buf_token_ids
        return buffers


__all__ = ["MiniMaxM2CompareForCausalLM"]
