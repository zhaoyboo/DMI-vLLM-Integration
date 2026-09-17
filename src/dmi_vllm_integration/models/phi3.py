# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from llama.py

# Adapted from vllm/model_executor/models/phi3.py in official vLLM 0.29.0.
"""Phi-3 causal LM using DMI's hooked Llama implementation."""

from dmi_vllm_integration.models.llama import LlamaPForCausalLM
from vllm.model_executor.models.phi3 import Phi3ForCausalLM as _Phi3ForCausalLM


class Phi3PForCausalLM(LlamaPForCausalLM):
    """Preserve Phi-3 fused weight packing while exposing Llama hooks."""

    packed_modules_mapping = _Phi3ForCausalLM.packed_modules_mapping


__all__ = ["Phi3PForCausalLM"]
