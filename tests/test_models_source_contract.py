"""Source-boundary checks for monitored models and validation copies."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


MODEL_DIRECTORY = (
    Path(__file__).parents[1] / "src" / "dmi_vllm_integration" / "models"
)
MODEL_CLASSES = {
    "apertus.py": "ApertusPForCausalLM",
    "deepseek_v4.py": "DeepseekV4PForCausalLM",
    "ernie45.py": "Ernie4_5PForCausalLM",
    "gemma3.py": "Gemma3PForCausalLM",
    "glm_moe_dsa.py": "GlmMoeDsaPForCausalLM",
    "gemma4.py": "Gemma4PForConditionalGeneration",
    "gpt2.py": "GPT2PLMHeadModel",
    "gpt_oss.py": "GptOssPForCausalLM",
    "granite.py": "GranitePForCausalLM",
    "kimi_k3.py": "KimiK3PForConditionalGeneration",
    "llama4.py": "Llama4PForCausalLM",
    "llama.py": "LlamaPForCausalLM",
    "mllama4.py": "Llama4PForConditionalGeneration",
    "minicpm.py": "MiniCPMPForCausalLM",
    "minimax_m2.py": "MiniMaxM2PForCausalLM",
    "mistral.py": "MistralPForCausalLM",
    "phi3.py": "Phi3PForCausalLM",
    "qwen2.py": "Qwen2PForCausalLM",
    "qwen2_moe.py": "Qwen2MoePForCausalLM",
    "qwen3.py": "Qwen3PForCausalLM",
    "qwen3_moe.py": "Qwen3MoePForCausalLM",
    "qwen3_5.py": "Qwen3_5PForConditionalGeneration",
}
UPSTREAM_SOURCE_PATHS = {
    "deepseek_v4.py": "vllm/models/deepseek_v4/nvidia/model.py",
    "kimi_k3.py": "vllm/models/kimi_k3/nvidia/model.py",
}
UPSTREAM_SOURCE_FILES = {
    "glm_moe_dsa.py": "deepseek_v2.py",
}
EXPANSION_MODEL_PROVENANCE = {
    "apertus.py": "vllm/model_executor/models/apertus.py",
    "deepseek_v4.py": "vllm/models/deepseek_v4/nvidia/model.py",
    "ernie45.py": "vllm/model_executor/models/ernie45.py",
    "gemma3.py": "vllm/model_executor/models/gemma3.py",
    "gemma4.py": "vllm/model_executor/models/gemma4.py",
    "glm_moe_dsa.py": "vllm/model_executor/models/deepseek_v2.py",
    "gpt_oss.py": "vllm/model_executor/models/gpt_oss.py",
    "granite.py": "vllm/model_executor/models/granite.py",
    "kimi_k3.py": "vllm/models/kimi_k3/nvidia/model.py",
    "llama4.py": "vllm/model_executor/models/llama4.py",
    "minicpm.py": "vllm/model_executor/models/minicpm.py",
    "minimax_m2.py": "vllm/model_executor/models/minimax_m2.py",
    "mistral.py": "vllm/model_executor/models/mistral.py",
    "mllama4.py": "vllm/model_executor/models/mllama4.py",
    "phi3.py": "vllm/model_executor/models/phi3.py",
    "qwen3_5.py": "vllm/model_executor/models/qwen3_5.py",
    "qwen3_moe.py": "vllm/model_executor/models/qwen3_moe.py",
}
EXPANSION_ORACLE_PROVENANCE = {
    "apertus_compare.py": EXPANSION_MODEL_PROVENANCE["apertus.py"],
    "deepseek_v4_compare.py": EXPANSION_MODEL_PROVENANCE["deepseek_v4.py"],
    "ernie45_compare.py": EXPANSION_MODEL_PROVENANCE["ernie45.py"],
    "gemma3_compare.py": EXPANSION_MODEL_PROVENANCE["gemma3.py"],
    "gemma4_compare.py": EXPANSION_MODEL_PROVENANCE["gemma4.py"],
    "glm_moe_dsa_compare.py": EXPANSION_MODEL_PROVENANCE["glm_moe_dsa.py"],
    "gpt_oss_compare.py": EXPANSION_MODEL_PROVENANCE["gpt_oss.py"],
    "granite_compare.py": EXPANSION_MODEL_PROVENANCE["granite.py"],
    "kimi_k3_compare.py": EXPANSION_MODEL_PROVENANCE["kimi_k3.py"],
    "minicpm_compare.py": EXPANSION_MODEL_PROVENANCE["minicpm.py"],
    "minimax_m2_compare.py": EXPANSION_MODEL_PROVENANCE["minimax_m2.py"],
    "mistral_compare.py": EXPANSION_MODEL_PROVENANCE["mistral.py"],
    "mllama4_compare.py": EXPANSION_MODEL_PROVENANCE["mllama4.py"],
    "phi3_compare.py": EXPANSION_MODEL_PROVENANCE["phi3.py"],
    "qwen3_5_compare.py": EXPANSION_MODEL_PROVENANCE["qwen3_5.py"],
    "qwen3_moe_compare.py": EXPANSION_MODEL_PROVENANCE["qwen3_moe.py"],
}
ORACLE_COPY_PROVENANCE = {
    "gpt2_compare.py": (
        "gpt2.py",
        "Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.",
    ),
    "gpt2_ref.py": (
        "gpt2.py",
        "Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.",
    ),
    "llama_compare.py": (
        "llama.py",
        "Copyright 2022 EleutherAI and the HuggingFace Inc. team.",
    ),
    "llama_ref.py": (
        "llama.py",
        "Copyright 2022 EleutherAI and the HuggingFace Inc. team.",
    ),
    "qwen3_ref.py": (
        "qwen3.py",
        "Copyright 2024 The Qwen team.",
    ),
}
ORACLE_DIRECTORY = Path(__file__).parent / "oracles"


def _installed_vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).parents[1]


def _leading_comment_header(path: Path) -> str:
    lines: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or (not line and lines):
            lines.append(line)
            continue
        break
    return "\n".join(lines).rstrip()


def _exact_upstream_prefix(upstream_path: str) -> str:
    upstream_header = _leading_comment_header(
        _installed_vllm_root() / upstream_path
    )
    return (
        f"{upstream_header}\n\n"
        f"# Adapted from {upstream_path} in official vLLM 0.29.0.\n"
    )


@pytest.mark.parametrize(("filename", "model_class"), MODEL_CLASSES.items())
def test_model_port_has_provenance_and_external_import_boundaries(
    filename: str,
    model_class: str,
) -> None:
    source = (MODEL_DIRECTORY / filename).read_text()
    tree = ast.parse(source)

    assert source.startswith("# SPDX-License-Identifier: Apache-2.0")
    upstream_path = UPSTREAM_SOURCE_PATHS.get(filename)
    if upstream_path is None:
        upstream_name = UPSTREAM_SOURCE_FILES.get(filename, filename)
        upstream_path = f"vllm/model_executor/models/{upstream_name}"
    assert f"Adapted from {upstream_path}" in source
    assert model_class in {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }

    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all(node.level == 0 for node in imports)
    dmi_api_imports = {
        node.module
        for node in imports
        if node.module is not None
        and (
            node.module == "dmi"
            or node.module.startswith(("dmi.", "monitoring"))
            or node.module == "dmi_vllm_integration.dmi_api"
        )
    }
    assert dmi_api_imports <= {"dmi_vllm_integration.dmi_api"}
    if "HookPoint" in source or "HookSpec" in source:
        assert dmi_api_imports == {"dmi_vllm_integration.dmi_api"}

    vllm_imports = {
        node.module
        for node in imports
        if node.module is not None and node.module.startswith("vllm")
    }
    assert any(
        module.startswith("vllm.model_executor.models")
        for module in vllm_imports
    )


@pytest.mark.parametrize(
    ("filename", "upstream_path"),
    EXPANSION_MODEL_PROVENANCE.items(),
)
def test_expansion_model_retains_exact_upstream_header(
    filename: str,
    upstream_path: str,
) -> None:
    source = (MODEL_DIRECTORY / filename).read_text()

    assert source.startswith(_exact_upstream_prefix(upstream_path))


@pytest.mark.parametrize(
    ("filename", "upstream_path"),
    EXPANSION_ORACLE_PROVENANCE.items(),
)
def test_expansion_oracle_retains_exact_upstream_header(
    filename: str,
    upstream_path: str,
) -> None:
    source = (ORACLE_DIRECTORY / filename).read_text()

    assert source.startswith(_exact_upstream_prefix(upstream_path))


@pytest.mark.parametrize(
    ("filename", "upstream_name", "upstream_copyright"),
    tuple(
        (filename, *provenance)
        for filename, provenance in ORACLE_COPY_PROVENANCE.items()
    ),
)
def test_copied_oracle_retains_upstream_license_and_provenance(
    filename: str,
    upstream_name: str,
    upstream_copyright: str,
) -> None:
    source = (ORACLE_DIRECTORY / filename).read_text()

    assert source.startswith("# SPDX-License-Identifier: Apache-2.0")
    assert (
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project"
        in source
    )
    assert upstream_copyright in source
    assert 'Licensed under the Apache License, Version 2.0 (the "License")' in source
    assert (
        f"Adapted from vllm/model_executor/models/{upstream_name} "
        "in official vLLM 0.29.0."
    ) in source


def test_qwen2_moe_activates_observer_patch_and_observes_one_route() -> None:
    source = (MODEL_DIRECTORY / "qwen2_moe.py").read_text()
    tree = ast.parse(source)

    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "apply_fused_moe_router_observer_patch" in calls
    assert "set_routing_observer" in source
    assert source.count(".select_experts(") == 0
