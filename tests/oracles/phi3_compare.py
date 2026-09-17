# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from llama.py

# Adapted from vllm/model_executor/models/phi3.py in official vLLM 0.29.0.
"""Phi-3 compare model for byte-identical DMI transport tests."""

from tests.oracles.llama_compare import LlamaCompareForCausalLM
from vllm.model_executor.models.phi3 import Phi3ForCausalLM as _Phi3ForCausalLM


class Phi3CompareForCausalLM(LlamaCompareForCausalLM):
    """Preserve Phi-3 packing in the independent-reference test model."""

    packed_modules_mapping = _Phi3ForCausalLM.packed_modules_mapping


__all__ = ["Phi3CompareForCausalLM"]
