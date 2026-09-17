# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from vllm/model_executor/models/mistral.py in official vLLM 0.29.0.
"""Bounded Mistral compare model for DMI transport-value tests."""

from vllm.config import VllmConfig

from tests.oracles.llama_compare import LlamaCompareForCausalLM
from dmi_vllm_integration.models.mistral import _reject_unsupported_mistral_branches


class MistralCompareForCausalLM(LlamaCompareForCausalLM):
    """Independent-buffer model for the audited Llama-equivalent cell."""

    embedding_modules: dict[str, str] = {}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _reject_unsupported_mistral_branches(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)


__all__ = ["MistralCompareForCausalLM"]
