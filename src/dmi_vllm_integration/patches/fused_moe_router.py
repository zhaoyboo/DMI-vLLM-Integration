"""Observe the one routing result consumed by modular vLLM MoE experts.

Official vLLM 0.29.0 has no public routing-observer hook. Monitored MoE models
need the exact router logits, selected weights, and selected IDs without
calling routing a second time, so this module wraps that release's
``select_experts`` boundary.
The patch is process-idempotent and is activated only when the monitored MoE
model module is imported.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from importlib.metadata import version
from typing import Any

from packaging.version import Version
import torch


SUPPORTED_VLLM_VERSIONS = frozenset({"0.29.0"})
_PATCH_VERSION = 2
_CLASS_PATCH_MARKER = "_dmi_routing_observer_patch_version"
_OBSERVER_ATTRIBUTE = "_dmi_routing_observer"

RoutingObserver = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    None,
]


def _installed_vllm_version() -> str:
    # ``public`` ignores local build metadata while retaining pre/dev/post
    # qualifiers, matching the integration's exact-release compatibility gate.
    return Version(version("vllm")).public


def _fused_moe_router_class() -> type[Any]:
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter,
    )

    return FusedMoERouter


def apply_fused_moe_router_observer_patch() -> bool:
    """Install the vLLM 0.29.0 routing observer wrapper.

    Returns ``True`` when this call installs the patch and ``False`` when the
    identical patch is already installed. Unsupported vLLM versions and a
    conflicting pre-existing observer API fail before any class is changed.
    """

    installed_version = _installed_vllm_version()
    if installed_version not in SUPPORTED_VLLM_VERSIONS:
        expected = ", ".join(sorted(SUPPORTED_VLLM_VERSIONS))
        raise RuntimeError(
            "DMI's FusedMoERouter observer patch supports vLLM "
            f"{expected}; found {installed_version}"
        )

    router_class = _fused_moe_router_class()
    installed_patch = getattr(router_class, _CLASS_PATCH_MARKER, None)
    if installed_patch == _PATCH_VERSION:
        return False
    if installed_patch is not None:
        raise RuntimeError(
            "FusedMoERouter has an incompatible DMI observer patch version: "
            f"{installed_patch}"
        )
    if "set_routing_observer" in router_class.__dict__:
        raise RuntimeError(
            "FusedMoERouter already defines set_routing_observer; expected the "
            "unmodified official vLLM implementation"
        )

    original_select_experts = router_class.select_experts

    def set_routing_observer(
        self: Any,
        observer: RoutingObserver | None,
    ) -> None:
        if observer is not None and not callable(observer):
            raise TypeError("routing observer must be callable or None")
        setattr(self, _OBSERVER_ATTRIBUTE, observer)

    @wraps(original_select_experts)
    def select_experts_with_observer(
        self: Any,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = original_select_experts(
            self,
            hidden_states,
            router_logits,
            topk_indices_dtype=topk_indices_dtype,
            input_ids=input_ids,
        )
        observer = getattr(self, _OBSERVER_ATTRIBUTE, None)
        if observer is not None:
            observer(router_logits, topk_weights, topk_ids)
        return topk_weights, topk_ids

    router_class.set_routing_observer = set_routing_observer
    router_class.select_experts = select_experts_with_observer
    setattr(router_class, _CLASS_PATCH_MARKER, _PATCH_VERSION)
    return True
