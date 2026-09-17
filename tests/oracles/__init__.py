"""Lazy registration for the integration suite's Compare/Ref models.

These models are validation oracles, not production plugin registrations.
Runners opt into them explicitly before constructing a vLLM engine.
"""

from __future__ import annotations


_ORACLE_MODELS = {
    "DMIApertusCompareForCausalLM": (
        "tests.oracles.apertus_compare:ApertusCompareForCausalLM"
    ),
    "DMIDeepseekV4CompareForCausalLM": (
        "tests.oracles.deepseek_v4_compare:DeepseekV4CompareForCausalLM"
    ),
    "DMIErnie4_5CompareForCausalLM": (
        "tests.oracles.ernie45_compare:Ernie4_5CompareForCausalLM"
    ),
    "DMIGemma3CompareForCausalLM": (
        "tests.oracles.gemma3_compare:Gemma3CompareForCausalLM"
    ),
    "DMIGlmMoeDsaCompareForCausalLM": (
        "tests.oracles.glm_moe_dsa_compare:GlmMoeDsaCompareForCausalLM"
    ),
    "DMIGemma4CompareForConditionalGeneration": (
        "tests.oracles.gemma4_compare:Gemma4CompareForConditionalGeneration"
    ),
    "DMIGPT2CompareLMHeadModel": "tests.oracles.gpt2_compare:GPT2CompareForCausalLM",
    "DMIGPT2RefLMHeadModel": "tests.oracles.gpt2_ref:GPT2RefLMHeadModel",
    "DMIGptOssCompareForCausalLM": (
        "tests.oracles.gpt_oss_compare:GptOssCompareForCausalLM"
    ),
    "DMIGraniteCompareForCausalLM": (
        "tests.oracles.granite_compare:GraniteCompareForCausalLM"
    ),
    "DMIKimiK3CompareForConditionalGeneration": (
        "tests.oracles.kimi_k3_compare:KimiK3CompareForConditionalGeneration"
    ),
    "DMILlamaCompareForCausalLM": "tests.oracles.llama_compare:LlamaCompareForCausalLM",
    "DMILlamaRefForCausalLM": "tests.oracles.llama_ref:LlamaRefForCausalLM",
    "DMILlama4CompareForConditionalGeneration": (
        "tests.oracles.mllama4_compare:Llama4CompareForConditionalGeneration"
    ),
    "DMIMiniCPMCompareForCausalLM": (
        "tests.oracles.minicpm_compare:MiniCPMCompareForCausalLM"
    ),
    "DMIMiniMaxM2CompareForCausalLM": (
        "tests.oracles.minimax_m2_compare:MiniMaxM2CompareForCausalLM"
    ),
    "DMIMistralCompareForCausalLM": (
        "tests.oracles.mistral_compare:MistralCompareForCausalLM"
    ),
    "DMIPhi3CompareForCausalLM": (
        "tests.oracles.phi3_compare:Phi3CompareForCausalLM"
    ),
    "DMIQwen2MoeCompareForCausalLM": (
        "tests.oracles.qwen2_moe_compare:Qwen2MoeCompareForCausalLM"
    ),
    "DMIQwen2MoeRefForCausalLM": (
        "tests.oracles.qwen2_moe_ref:Qwen2MoeRefForCausalLM"
    ),
    "DMIQwen3CompareForCausalLM": (
        "tests.oracles.qwen3_compare:Qwen3CompareForCausalLM"
    ),
    "DMIQwen3MoeCompareForCausalLM": (
        "tests.oracles.qwen3_moe_compare:Qwen3MoeCompareForCausalLM"
    ),
    "DMIQwen3_5CompareForConditionalGeneration": (
        "tests.oracles.qwen3_5_compare:Qwen3_5CompareForConditionalGeneration"
    ),
    "DMIQwen3RefForCausalLM": "tests.oracles.qwen3_ref:Qwen3RefForCausalLM",
}


def register_oracle_models() -> None:
    """Register validation-only architectures without importing model code."""

    from vllm import ModelRegistry

    supported = ModelRegistry.get_supported_archs()
    for architecture, model_path in _ORACLE_MODELS.items():
        if architecture not in supported:
            ModelRegistry.register_model(architecture, model_path)


__all__ = ["register_oracle_models"]
