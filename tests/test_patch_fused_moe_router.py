"""Focused tests for the version-gated FusedMoERouter observer patch."""

from __future__ import annotations

import pytest

from dmi_vllm_integration.patches import fused_moe_router


def _patch_fake_router(
    monkeypatch: pytest.MonkeyPatch,
    router_class: type,
    *,
    version: str = "0.29.0",
) -> None:
    monkeypatch.setattr(
        fused_moe_router,
        "_installed_vllm_version",
        lambda: version,
    )
    monkeypatch.setattr(
        fused_moe_router,
        "_fused_moe_router_class",
        lambda: router_class,
    )


def test_patch_observes_the_single_returned_route_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRouter:
        def __init__(self) -> None:
            self.calls = 0

        def select_experts(
            self,
            hidden_states,
            router_logits,
            topk_indices_dtype=None,
            *,
            input_ids=None,
        ):
            del hidden_states, router_logits, topk_indices_dtype, input_ids
            self.calls += 1
            return object(), object()

    _patch_fake_router(monkeypatch, FakeRouter)
    assert fused_moe_router.apply_fused_moe_router_observer_patch() is True
    assert fused_moe_router.apply_fused_moe_router_observer_patch() is False

    router = FakeRouter()
    observed: list[tuple[object, object, object]] = []
    hidden_states = object()
    router_logits = object()
    router.set_routing_observer(
        lambda logits, weights, ids: observed.append((logits, weights, ids))
    )
    returned = router.select_experts(hidden_states, router_logits)

    assert router.calls == 1
    assert observed == [(router_logits, *returned)]

    router.set_routing_observer(None)
    second = router.select_experts(object(), object())
    assert router.calls == 2
    assert observed == [(router_logits, *returned)]
    assert second != returned


@pytest.mark.parametrize("installed_version", ["0.29.0rc1", "0.29.0.post1", "0.28.0"])
def test_patch_rejects_unsupported_vllm_before_importing_router(
    monkeypatch: pytest.MonkeyPatch,
    installed_version: str,
) -> None:
    monkeypatch.setattr(
        fused_moe_router,
        "_installed_vllm_version",
        lambda: installed_version,
    )
    monkeypatch.setattr(
        fused_moe_router,
        "_fused_moe_router_class",
        lambda: pytest.fail("router class must not be imported"),
    )

    with pytest.raises(RuntimeError, match=r"supports vLLM 0\.29\.0"):
        fused_moe_router.apply_fused_moe_router_observer_patch()


def test_version_gate_accepts_local_build_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_router, "version", lambda name: "0.29.0+cu130")
    assert fused_moe_router._installed_vllm_version() == "0.29.0"


def test_patch_rejects_a_conflicting_observer_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConflictingRouter:
        def set_routing_observer(self, observer) -> None:
            del observer

        def select_experts(self, *args, **kwargs):
            del args, kwargs
            return object(), object()

    original = ConflictingRouter.select_experts
    _patch_fake_router(monkeypatch, ConflictingRouter)

    with pytest.raises(RuntimeError, match="unmodified official vLLM"):
        fused_moe_router.apply_fused_moe_router_observer_patch()
    assert ConflictingRouter.select_experts is original


def test_patch_validates_observer_value(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRouter:
        def select_experts(self, *args, **kwargs):
            del args, kwargs
            return object(), object()

    _patch_fake_router(monkeypatch, FakeRouter)
    fused_moe_router.apply_fused_moe_router_observer_patch()

    with pytest.raises(TypeError, match="callable or None"):
        FakeRouter().set_routing_observer(object())
