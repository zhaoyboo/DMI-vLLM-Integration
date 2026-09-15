"""vLLM integration: VLLMAdaptor + monitored GPU worker.

Phase 3a of the unified-adaptor refactor consolidates the vLLM-specific
orchestration that used to live in ``monitoring/vllm_integration.py``
into one file under ``integration/``.

Key pieces:

  * ``VLLMAdaptor`` -- concrete ``BackendAdaptor`` for vLLM models.
    Owns the framework-fragile pieces in localized methods:
      ``_record_real_layout`` (copies post-prepare packed request order for
      either model runner);
      ``_preflight_force_eager`` (evaluates cached worst-role formulas);
      ``build_step_context`` (uses the real layout and batch descriptor);
      ``adapt_for_cpu_direct`` (swaps padded -> unpadded q_len when an
      oversize step forces eager dispatch + safety net for this batch);
      ``before_forward`` (commits one already-computed actual plan);
      ``_warn_once_capacity`` (per-(total_q, num_reqs) shape warn).
  * ``DMXGPUWorker`` -- vLLM ``Worker`` subclass that owns a
    ``VLLMAdaptor``, records the real input layout, and commits after the real
    dispatch but before model forward. Architecture remap stays here.
  * Module-level ``register_preset("vllm-full", ...)`` -- relocated
    from ``monitoring/selection.py``'s default ``_HOOK_SELECTIONS``
    (deferred from Phase 1.5 per the unified-adaptor plan).  Lands as
    a side-effect of importing this module.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import re
import warnings
from typing import Any, List, Optional, Tuple
from uuid import uuid4
from weakref import WeakSet

import numpy as np
import torch

from vllm import LLM
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.worker.gpu_worker import Worker

from ..architectures import (
    ARCHITECTURE_REMAP as _ARCH_REMAP,
    require_supported_architecture,
)
from ..compat import require_compatible_runtime
from ..model_shape import make_model_shape_from_hf_config
from ..model_validation import validate_model_specific_config

from dmi_vllm_integration.dmi_api import (
    ALL_HOOK_TYPES,
    ATTENTION_WEIGHT_HOOK_TYPES,
    BackendAdaptor,
    CHClickhouseDriverReadOnly,
    ClickHouseClientConfig,
    DMXHostEngine,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    HookRowBasis,
    HookSpec,
    ModelShapeConfig,
    MonitoringEngine,
    RingConfig,
    StageConfig,
    StepContext,
    align_up,
    compute_hook_shape,
    configure_hook_padding_strip,
    hook_row_basis,
    hook_belongs_to_pp_rank,
    hook_belongs_to_tp_rank,
    is_preset_registered,
    make_lazy_internal,
    register_preset,
    select_hook_specs,
)


require_compatible_runtime()


# ---------------------------------------------------------------------------
# vLLM-full preset registration (deferred from Phase 1.5).
#
# Moved out of monitoring/selection.py's default _HOOK_SELECTIONS so the
# core selection module is framework-neutral.  Registers when this
# module is imported -- which happens whenever DMXGPUWorker is loaded
# via worker_cls="dmi_vllm_integration.worker.DMXGPUWorker".
#
# `register_preset` raises on duplicates, so re-import within the same
# process is a no-op (Python caches the module body).  Across separate
# processes (e.g. each TP rank in vLLM) each subprocess imports fresh
# and registers once.
# ---------------------------------------------------------------------------

if not is_preset_registered("vllm-full"):
    register_preset(
        "vllm-full",
        ALL_HOOK_TYPES - ATTENTION_WEIGHT_HOOK_TYPES,
    )


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _cfg(ac: dict, key: str, env_key: str, default: Any) -> Any:
    """additional_config preferred, env var fallback, then default."""
    val = ac.get(key)
    if val is not None:
        return val
    env_val = os.environ.get(env_key)
    if env_val is not None:
        if isinstance(default, bool):
            return env_val not in ("0", "false", "False", "")
        if isinstance(default, int):
            return int(env_val)
        return env_val
    return default


def _resolve_hook_types(selection: str) -> frozenset[int]:
    """Resolve one public hook-selection expression to hook-type IDs."""
    return frozenset(
        spec.hook_type
        for spec in select_hook_specs(
            [HookSpec(hook_type, None) for hook_type in ALL_HOOK_TYPES],
            selection,
        )
    )


def _resolve_hook_selection_tokens(
    selection: str,
) -> tuple[frozenset[int], tuple[tuple[str, frozenset[int]], ...]]:
    """Resolve a selection while preserving comma-token provenance."""
    selected = _resolve_hook_types(selection)
    tokens = tuple(
        token.strip() for token in selection.split(",") if token.strip()
    )
    return selected, tuple((token, _resolve_hook_types(token)) for token in tokens)


def _token_ids_unavailability_reason(model_config: Any) -> Optional[str]:
    """Explain why vLLM V1 will not pass ``input_ids`` to this model."""
    if bool(getattr(model_config, "enable_prompt_embeds", False)):
        return (
            "prompt embeddings are enabled, so vLLM passes inputs_embeds "
            "instead of input_ids"
        )

    # Mirror GPUModelRunner._preprocess. Multimodal-capable models whose
    # modality limits are all disabled take vLLM's text-only branch and still
    # receive IDs, so ModelConfig.is_multimodal_model alone is insufficient.
    supports_mm_inputs = (
        bool(getattr(model_config, "is_multimodal_model", False))
        and MULTIMODAL_REGISTRY.supports_multimodal_inputs(model_config)
    )
    if (
        supports_mm_inputs
        and not bool(getattr(model_config, "is_encoder_decoder", False))
        and not bool(getattr(model_config, "requires_raw_input_tokens", False))
    ):
        return (
            "vLLM's active multimodal input path passes inputs_embeds without "
            "input_ids because the model does not require raw input tokens"
        )
    return None


def _effective_selection_preset(hook_types: frozenset[int]) -> str:
    """Return a deterministic internal preset for an effective hook set."""
    encoded_types = "_".join(str(hook_type) for hook_type in sorted(hook_types))
    name = f"__dmi_vllm_effective_without_token_ids_{encoded_types}"
    if not is_preset_registered(name):
        register_preset(name, hook_types)
    elif _resolve_hook_types(name) != hook_types:
        raise RuntimeError(
            "DMI vLLM's reserved effective hook-selection preset was "
            f"registered with conflicting hook types: {name!r}"
        )
    return name


_VLLM_REQ_ID_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def normalize_vllm_request_id(req_id: str) -> str:
    """Return the external ID encoded in a V1 scheduler request ID.

    vLLM only appends the eight-character suffix when request-ID
    randomization is enabled. An external ID that happens to end in the same
    pattern must remain untouched when randomization is disabled.
    """
    from vllm import envs

    if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
        return req_id
    return _VLLM_REQ_ID_SUFFIX.sub("", req_id)


# ---------------------------------------------------------------------------
# Per-call request-layout / dispatch state
# ---------------------------------------------------------------------------


class VLLMStepPhase(Enum):
    IDLE = auto()
    ARMED = auto()
    LAYOUT_READY = auto()
    COMMITTED = auto()


class VLLMValidationMode(Enum):
    OFF = auto()
    VERIFY = auto()


@dataclass(frozen=True)
class _VLLMRealLayout:
    raw_req_ids: Tuple[str, ...]
    req_ids: Tuple[str, ...]
    scheduled_counts: Tuple[int, ...]
    computed_counts: Tuple[int, ...]
    token_ranges: Tuple[Tuple[int, int], ...]
    dim0_offsets: Tuple[int, ...]
    total_rows: int


@dataclass
class _VLLMStepState:
    phase: VLLMStepPhase = VLLMStepPhase.IDLE
    scheduler_output: Any = None
    layout: Optional[_VLLMRealLayout] = None
    force_eager_latch: bool = False
    prelayout_dispatch_seen: bool = False
    capacity_candidate: Optional[Tuple[Any, Any]] = None
    expected_dispatch: Optional[Tuple[Any, Any]] = None
    execution_bound: Optional[int] = None
    real_rows_bound: Optional[int] = None
    request_rows_bound: Optional[int] = None
    real_cudagraph_mode: Any = None
    real_batch_descriptor: Any = None


@dataclass(frozen=True)
class _VLLMHookSelection:
    local_hooks: Tuple[HookSpec, ...]
    candidate_rank_hook_sets: Tuple[Tuple[HookSpec, ...], ...]
    selected_hook_types: frozenset[int]

    @classmethod
    def from_model(
        cls,
        *,
        model: Any,
        local_hooks: Tuple[HookSpec, ...],
        hook_selection: str,
        cfg: ModelShapeConfig,
        parallel_config: Any,
        hf_config: Any,
        final_logits_dtype: Optional[torch.dtype] = None,
    ) -> "_VLLMHookSelection":
        """Select local and model-wide rank-type hooks once at attachment."""
        hf_config = getattr(hf_config, "text_config", hf_config)
        tp_size = int(parallel_config.tensor_parallel_size)
        pp_size = int(parallel_config.pipeline_parallel_size)
        if tp_size < 1 or pp_size < 1:
            raise RuntimeError(
                f"Invalid vLLM parallel sizes: TP={tp_size}, PP={pp_size}"
            )

        num_layers = getattr(
            hf_config,
            "num_hidden_layers",
            getattr(hf_config, "n_layer", None),
        )
        if num_layers is None:
            raise RuntimeError("DMI vLLM model has no hidden-layer count")
        num_layers = int(num_layers)

        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        selection_model = (
            model.unwrap()
            if isinstance(model, CUDAGraphWrapper)
            else model
        )
        get_hook_specs = getattr(selection_model, "get_hook_specs", None)
        if get_hook_specs is None:
            raise RuntimeError(
                "DMI vLLM model does not provide get_hook_specs"
            )
        try:
            model_wide_specs = get_hook_specs(model_wide=True)
        except TypeError as exc:
            raise RuntimeError(
                "DMI vLLM model does not support "
                "get_hook_specs(model_wide=True)"
            ) from exc
        for spec in model_wide_specs:
            if (
                spec.hook_type == HOOK_TYPE_FINAL_LOGITS
                and final_logits_dtype is not None
            ):
                spec.dtype = final_logits_dtype
        if any(spec.module is not None for spec in model_wide_specs):
            raise RuntimeError(
                "DMI vLLM model-wide HookSpecs must not bind live modules"
            )
        inventory_layers = {
            spec.layer_no for spec in model_wide_specs if spec.layer_no >= 0
        }
        if inventory_layers != set(range(num_layers)):
            raise RuntimeError(
                "DMI vLLM model-wide hook inventory does not cover every layer"
            )

        selected_model_wide = select_hook_specs(
            model_wide_specs, hook_selection, cfg
        )

        from vllm.distributed.utils import get_pp_indices

        pp_intervals = tuple(
            get_pp_indices(num_layers, pp_rank, pp_size)
            for pp_rank in range(pp_size)
        )
        covered_layers = [
            layer
            for start, end in pp_intervals
            for layer in range(start, end)
        ]
        if covered_layers != list(range(num_layers)):
            raise RuntimeError("vLLM PP layer partition is not an exact cover")

        tp_role_count = min(tp_size, 2)
        candidate_rank_hook_sets: List[Tuple[HookSpec, ...]] = []
        for pp_rank, (start_layer, end_layer) in enumerate(pp_intervals):
            pp_specs = tuple(
                spec
                for spec in selected_model_wide
                if (
                    (
                        spec.layer_no < 0
                        or start_layer <= spec.layer_no < end_layer
                    )
                    and hook_belongs_to_pp_rank(
                        spec,
                        is_first_rank=(pp_rank == 0),
                        is_last_rank=(pp_rank == pp_size - 1),
                    )
                )
            )
            for tp_role in range(tp_role_count):
                candidate_rank_hook_sets.append(
                    tuple(
                        spec
                        for spec in pp_specs
                        if hook_belongs_to_tp_rank(spec, tp_role)
                    )
                )

        return cls(
            local_hooks=local_hooks,
            candidate_rank_hook_sets=tuple(candidate_rank_hook_sets),
            selected_hook_types=frozenset(
                spec.hook_type for spec in selected_model_wide
            ),
        )


@dataclass(frozen=True)
class _VLLMRoleFormula:
    real_terms: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    execution_terms: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    request_terms: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    hook_count: int = 0

    def bytes_for(
        self, execution_rows: int, real_rows: int, request_rows: int
    ) -> int:
        total = 0
        for row_bytes, multiplicity in self.real_terms:
            total += multiplicity * align_up(real_rows * row_bytes, 16)
        for row_bytes, multiplicity in self.execution_terms:
            total += multiplicity * align_up(execution_rows * row_bytes, 16)
        for row_bytes, multiplicity in self.request_terms:
            total += multiplicity * align_up(request_rows * row_bytes, 16)
        return total


# ---------------------------------------------------------------------------
# VLLMAdaptor
# ---------------------------------------------------------------------------


class VLLMAdaptor(BackendAdaptor):
    """``BackendAdaptor`` for vLLM models running under ``DMXGPUWorker``.

    vLLM's per-step protocol differs from HF's:
      * Tensors are packed/flattened (no batch dim; rows from each
        request are concatenated along dim 0).
      * Request IDs come from ``scheduler_output``, not auto-minted.
      * CUDA-graph dispatch may pad ``total_q`` up to the nearest
        capture size; meta shape must match the padded tensor.
      * Cached role formulas decide whether the current dispatch must be
        eager before vLLM selects its real batch descriptor.
      * Reservation and metadata use that real descriptor plus the packed
        request order recorded after vLLM prepares its inputs.
    """

    def __init__(
        self, engine: Any, model_id: str, vllm_config: Any,
        *, gpu_padding_strip: bool = True,
    ) -> None:
        super().__init__(engine, model_id)
        self.vllm_config = vllm_config
        self._debug_step: bool = bool(os.environ.get("RING_DEBUG_STEP"))
        # The per-call state carries a monotonic pre-dispatch eager latch.
        # transport.force_eager is assigned during the actual-layout commit
        # for the producer path of that same model step.
        # User opt-in; when True, ring transport stays in null mode after
        # warmup (kernels fire as no-ops, FIFO stays empty).
        self.user_wants_null_mode: bool = False
        # Stashed during build_step_context so adapt_for_cpu_direct can
        # restore the unpadded q_len without recomputing.
        self._last_total_q: int = 0
        # vLLM step counter (debug logging only).
        self._step_counter: int = 0
        # gpu_padding_strip mode: when True, the producer copies only
        # actual_q_len * row_bytes per eligible hook (instead of the
        # full padded captured tensor).  Default False = today's
        # behavior verbatim.  When True, attach_model allocates the
        # shared row_count tensor pair and wires each eligible
        # HookPoint's _strip_tensor + _strip_row_bytes; build_step_context
        # populates ctx.actual_q_len; before_forward does the per-step
        # 8-byte cudaMemcpyAsync of the new actual_q_len.
        self.gpu_padding_strip: bool = gpu_padding_strip
        self._row_count_dev: Optional[torch.Tensor] = None      # 1 int64 on GPU
        self._pinned_row_count: Optional[torch.Tensor] = None   # 1 int64 pinned host
        self._strip_eligible_hps: List[Any] = []                # for inspection
        self._step_state = _VLLMStepState()
        # Integration-owned snapshot for validation workers.  Keeping this
        # here avoids exposing DMI's transport or its mutable metadata state.
        self._last_committed_layout: Optional[_VLLMRealLayout] = None
        self._validation_mode = (
            VLLMValidationMode.VERIFY
            if self._debug_step
            else VLLMValidationMode.OFF
        )
        self._role_formulas: Tuple[_VLLMRoleFormula, ...] = ()
        self._local_role_formula: Optional[_VLLMRoleFormula] = None
        self._has_global_hooks = False
        self._captures_final_logits = False
        self._byte_capacity = 0
        self._task_capacity = 0
        self._max_num_batched_tokens = 0
        self._max_num_seqs = 0
        self._capture_sizes: Tuple[int, ...] = ()
        self._max_capture_size = 0
        self._capacity_warned_shapes: set[tuple[int, int]] = set()

    @property
    def last_committed_layout(self) -> Optional[_VLLMRealLayout]:
        """Return the immutable packed layout from the last committed step."""

        return self._last_committed_layout

    # ------- abstract overrides ---------------------------------------

    def detect_model_shape(self, model: Any) -> ModelShapeConfig:
        # Use the vllm_config dtype (authoritative); fall back to the
        # HF config's torch_dtype if vllm_config is missing.
        vllm_dtype = getattr(
            getattr(self.vllm_config, "model_config", None), "dtype", None
        )
        cfg = make_model_shape_from_hf_config(
            self.vllm_config.model_config.hf_config, dtype=vllm_dtype
        )
        if cfg is None:
            raise RuntimeError(
                "VLLMAdaptor.detect_model_shape: vllm_config.model_config.hf_config "
                "is missing hidden_size / num_attention_heads."
            )
        # vLLM TP world size comes from its parallel-state group (NOT
        # torch.distributed; vLLM constructs its own TP/PP groups).
        # ``_compute_hook_shape`` divides head/intermediate dims by
        # ``cfg.tp_size`` for sharded hooks (q, k, v, z, mlp_post,
        # attn_scores, pattern) -- so getting this right is load-bearing
        # for TP > 1: the producer-side meta shape must match the actual
        # sharded tensor, otherwise the drain thread rejects the row.
        from vllm.distributed.parallel_state import get_tp_group
        cfg.tp_size = max(1, get_tp_group().world_size)
        return cfg

    def detect_parallel_ranks(self) -> Tuple[int, int, int, int]:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        tp_rank = get_tp_group().rank_in_group
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        ep_rank = 0  # vLLM does not expose EP groups today; placeholder.
        pp_rank = get_pp_group().rank_in_group
        return (tp_rank, dp_rank, ep_rank, pp_rank)

    def is_pp_first(self) -> bool:
        from vllm.distributed.parallel_state import get_pp_group
        return get_pp_group().is_first_rank

    def is_pp_last(self) -> bool:
        from vllm.distributed.parallel_state import get_pp_group
        return get_pp_group().is_last_rank

    def on_capacity_exceeded(self, ctx: StepContext) -> None:
        # No-op.  transport.force_eager is owned by adaptor_base
        # before_forward (`force_eager = (result == 2) or needs_eager`).
        # Kept as a framework hook for subclasses that want to react to
        # overflow events (telemetry, custom logging, etc.).
        return

    def adapt_for_cpu_direct(self, ctx: StepContext) -> StepContext:
        # When the worker forces eager for this batch, the actual
        # tensor will not be padded -- swap meta q_len from padded
        # back to the unpadded total_q stashed during build_step_context.
        import dataclasses
        if self._last_total_q and ctx.q_len != self._last_total_q:
            return dataclasses.replace(ctx, q_len=self._last_total_q)
        return ctx

    def _warn_once_capacity(
        self, ctx: StepContext, total_bytes: int, n_hooks: int
    ) -> None:
        # Per-(total_q, num_reqs) shape warn.  Stored on adapter (not
        # transport) so multiple adapters in the same process don't
        # share state.
        shape_key = (ctx.q_len, len(ctx.req_ids))
        if shape_key in self._capacity_warned_shapes:
            return
        self._capacity_warned_shapes.add(shape_key)
        try:
            capacities = self.engine.ring_capacities()
        except RuntimeError:
            return
        pcap = capacities.payload_bytes
        scap = capacities.staging_bytes
        if total_bytes > pcap and total_bytes > scap:
            reason = (
                f"exceeds both GPU ring ({pcap / 1e6:.0f} MB) "
                f"and pinned staging ({scap / 1e6:.0f} MB)"
            )
        elif total_bytes > pcap:
            reason = f"exceeds GPU ring ({pcap / 1e6:.0f} MB)"
        else:
            reason = f"exceeds pinned staging ({scap / 1e6:.0f} MB)"
        warnings.warn(
            f"[vllm_integration] Step data ({total_bytes / 1e6:.1f} MB) "
            f"{reason}. Falling back to eager dispatch + per-hook safety net "
            f"for {n_hooks} hooks.",
            stacklevel=2,
        )

    # ------- vLLM-specific helpers ------------------------------------

    def predict_padded_q_len(self, model_runner: Any, total_q: int) -> int:
        """Read vLLM's CUDA-graph capture size table and return the
        padded ``q_len`` the producer kernel will see.

        Risk surface: this reads ``cudagraph_dispatcher._bs_to_padded_graph_size``,
        a private list[int] indexed by token count.  On a vLLM upgrade,
        verify the attribute still exists and the per-step dispatch
        logic hasn't added new conditions that bypass CUDA graphs (LoRA,
        cascade, encoder, etc.).  Today the only per-step disabler is
        our own ``force_eager_next_batch``.
        """
        pad_table = getattr(
            getattr(model_runner, "cudagraph_dispatcher", None),
            "_bs_to_padded_graph_size", None,
        )
        if pad_table is not None and total_q < len(pad_table):
            return pad_table[total_q]
        return total_q

    @staticmethod
    def _input_ids_dtype(model_runner: Any) -> Optional[torch.dtype]:
        """Return the runner-owned token buffer dtype without a device copy."""
        input_ids = getattr(model_runner, "input_ids", None)
        if input_ids is not None:
            gpu = getattr(input_ids, "gpu", None)
            return getattr(gpu, "dtype", None)

        input_buffers = getattr(model_runner, "input_buffers", None)
        return getattr(getattr(input_buffers, "input_ids", None), "dtype", None)

    def build_step_context(
        self, scheduler_output: Any, model_runner: Any
    ) -> Optional[StepContext]:
        """Build metadata from the real packed layout and real dispatch."""
        state = self._step_state
        layout = state.layout
        batch_descriptor = state.real_batch_descriptor
        if layout is None or batch_descriptor is None:
            raise RuntimeError(
                "DMI vLLM step context requested before real layout/dispatch"
            )
        if state.scheduler_output is not scheduler_output:
            raise RuntimeError("DMI vLLM scheduler output changed within one step")
        if layout.total_rows == 0:
            return None

        self._step_counter += 1
        _step = self._step_counter
        self._last_total_q = layout.total_rows

        if self._debug_step:
            for i, req_id in enumerate(layout.req_ids):
                start, end = layout.token_ranges[i]
                print(
                    f"[dmx_worker] step={_step} req[{i}] rid={req_id} "
                    f"offset={layout.dim0_offsets[i]} "
                    f"n={layout.scheduled_counts[i]} "
                    f"t_start={start} t_end={end} "
                    f"pre_computed={layout.computed_counts[i]} "
                    f"q_len={batch_descriptor.num_tokens}",
                    flush=True,
                )

        # Read the V1 or V2 runner-owned input buffer without copying it. On
        # non-first PP ranks the buffer may not exist; token_ids is already
        # filtered out by PP placement there.
        ids_dtype = self._input_ids_dtype(model_runner)

        tp_rank, dp_rank, ep_rank, pp_rank = self.detect_parallel_ranks()

        # logits_to_keep=num_reqs: vLLM's compute_logits returns one
        # logit per request shaped [num_reqs, vocab].  In flattened
        # mode _compute_hook_shape uses logits_to_keep directly as
        # dim0 (no batch dim), so the meta shape becomes
        # [num_reqs, vocab].  The p2p thread then slices row j for
        # request j and adjusts the DB token range to (end_token-1,
        # end_token) -- the single predicted position per request.
        return StepContext(
            model_id=str(self.model_id),
            flattened=True,
            req_ids=list(layout.req_ids),
            token_ranges=list(layout.token_ranges),
            dim0_offsets=list(layout.dim0_offsets),
            kv_offsets=[0] * len(layout.req_ids),
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            ep_rank=ep_rank,
            pp_rank=pp_rank,
            batch=0,
            q_len=int(batch_descriptor.num_tokens),
            kv_dim=0,
            logits_to_keep=len(layout.req_ids),
            token_ids_dtype=ids_dtype,
            actual_q_len=(
                layout.total_rows if self.gpu_padding_strip else None
            ),
        )

    # ----- gpu_padding_strip integration -----

    def attach_model(self, model: Any, hook_selection: str = "full") -> None:
        """Standard attach, plus gpu_padding_strip pool setup when enabled.

        When `gpu_padding_strip=True`, allocate one shared int64[1] device
        tensor (`_row_count_dev`) and one pinned-host counterpart.  For
        every active spec with `dim0_is_actual_tokens=True`, point the
        HookPoint's `_strip_tensor` at the shared tensor and bake its
        per-spec `_strip_row_bytes` (CPU-known constant).  Every step
        we update the shared tensor's value in place (see
        before_forward); each HookPoint's captured producer_prefix
        call reads the freshly-written value and multiplies by its
        baked row_bytes.
        """
        super().attach_model(model, hook_selection)
        head_dtype = self.vllm_config.model_config.head_dtype
        for spec in self.active_hook_specs:
            if spec.hook_type == HOOK_TYPE_FINAL_LOGITS:
                spec.dtype = head_dtype
        if self.gpu_padding_strip:
            configured_device = getattr(
                getattr(self.vllm_config, "device_config", None),
                "device",
                None,
            )
            if configured_device is None:
                raise RuntimeError(
                    "DMI vLLM padding-strip setup requires "
                    "vllm_config.device_config.device"
                )
            try:
                device = torch.device(configured_device)
            except (TypeError, RuntimeError) as exc:
                raise RuntimeError(
                    "DMI vLLM received an invalid configured device for "
                    f"padding-strip setup: {configured_device!r}"
                ) from exc
            if device.type != "cuda":
                raise RuntimeError(
                    "DMI vLLM padding-strip setup requires a CUDA device; "
                    f"configured device is {device}"
                )
            self._row_count_dev = torch.empty(
                1, dtype=torch.int64, device=device
            )
            self._pinned_row_count = torch.empty(
                1, dtype=torch.int64, pin_memory=True
            )
            active_specs = self.active_hook_specs
            model_cfg = self.model_shape
            if model_cfg is None:
                raise RuntimeError("DMI vLLM model shape is unavailable")
            for spec in active_specs:
                if not spec.dim0_is_actual_tokens:
                    continue
                hp = spec.module
                rb = _row_bytes_for_spec(spec, model_cfg)
                if rb <= 0:
                    continue
                configure_hook_padding_strip(hp, self._row_count_dev, rb)
                self._strip_eligible_hps.append(hp)

        active_specs = self.active_hook_specs
        cfg = self.model_shape
        if cfg is None:
            raise RuntimeError("DMI vLLM role formulas require an attached ring")
        selection = _VLLMHookSelection.from_model(
            model=model,
            local_hooks=active_specs,
            hook_selection=hook_selection,
            cfg=cfg,
            parallel_config=self.vllm_config.parallel_config,
            hf_config=self.vllm_config.model_config.hf_config,
            final_logits_dtype=head_dtype,
        )
        if not self.user_wants_null_mode:
            self._validate_moe_routing_capture(
                model,
                active_specs,
                self.vllm_config.parallel_config,
                selected_hook_types=selection.selected_hook_types,
            )
        self._compile_role_formulas(selection)

    @staticmethod
    def _validate_moe_routing_capture(
        model: Any,
        specs: Tuple[HookSpec, ...],
        parallel_config: Any = None,
        *,
        selected_hook_types: Optional[frozenset[int]] = None,
    ) -> None:
        """Reject selected routing hooks on incompatible MoE execution paths."""
        if selected_hook_types is None:
            selected_hook_types = frozenset(spec.hook_type for spec in specs)
        selected_captures_topk = any(
            hook_type in (HOOK_TYPE_TOPK_IDS, HOOK_TYPE_TOPK_WEIGHTS)
            for hook_type in selected_hook_types
        )
        selected_captures_topk_ids = any(
            hook_type == HOOK_TYPE_TOPK_IDS for hook_type in selected_hook_types
        )
        selected_captures_router_logits = any(
            hook_type == HOOK_TYPE_ROUTER_LOGITS for hook_type in selected_hook_types
        )
        sp_unsupported_hook_types = frozenset(
            getattr(model, "dmi_sequence_parallel_unsupported_hook_types", ())
        )
        if (
            parallel_config is not None
            and getattr(parallel_config, "use_sequence_parallel_moe", False)
            and selected_hook_types & sp_unsupported_hook_types
        ):
            raise RuntimeError(
                "DMI selected hooks require a model forward that does not "
                "support sequence-parallel MoE"
            )
        if (
            parallel_config is not None
            and getattr(parallel_config, "use_sequence_parallel_moe", False)
            and (selected_captures_topk or selected_captures_router_logits)
        ):
            raise RuntimeError(
                "DMI routing-hook capture does not support sequence-parallel "
                "MoE token shards"
            )
        if (
            selected_captures_topk_ids
            and parallel_config is not None
            and getattr(parallel_config, "enable_eplb", False)
        ):
            raise RuntimeError(
                "DMI top-k ID capture requires logical expert IDs and does "
                "not support EPLB physical expert remapping"
            )

        captures_topk = any(
            spec.hook_type in (HOOK_TYPE_TOPK_IDS, HOOK_TYPE_TOPK_WEIGHTS)
            for spec in specs
        )
        captures_router_logits = any(
            spec.hook_type == HOOK_TYPE_ROUTER_LOGITS for spec in specs
        )
        if not captures_topk and not captures_router_logits:
            return

        from vllm.model_executor.layers.fused_moe.layer import MoERunner

        moe_runners = tuple(
            module for module in model.modules()
            if isinstance(module, MoERunner)
        )
        if not moe_runners:
            raise RuntimeError(
                "DMI selected top-k routing hooks but found no local MoE runner"
            )
        if captures_topk and any(runner.is_monolithic for runner in moe_runners):
            raise RuntimeError(
                "DMI top-k routing capture requires a modular MoE backend; "
                "select one with --moe-backend (for example, triton)."
            )
        if captures_router_logits and any(
            runner.is_monolithic and runner.gate is not None
            for runner in moe_runners
        ):
            raise RuntimeError(
                "DMI router-logit capture requires a modular MoE backend "
                "when vLLM owns the model's router"
            )
        if captures_topk and any(
            runner.do_naive_dispatch_combine
            or (
                runner.moe_config.pcp_size > 1
                and not runner.moe_config.moe_parallel_config.use_all2all_kernels
            )
            for runner in moe_runners
        ):
            raise RuntimeError(
                "DMI top-k routing capture does not support vLLM paths that "
                "dispatch cross-rank token rows before routing"
            )

    def _compile_role_formulas(
        self, selection: _VLLMHookSelection
    ) -> None:
        """Compile exact byte/hook formulas for distinct TP/PP rank types."""
        cfg = self.model_shape
        if cfg is None:
            raise RuntimeError("DMI vLLM role formulas require an attached ring")

        parallel = self.vllm_config.parallel_config
        scheduler = self.vllm_config.scheduler_config
        tp_size = int(parallel.tensor_parallel_size)
        if parallel.data_parallel_size > 1 and parallel.use_ubatching:
            raise RuntimeError(
                "DMI vLLM does not support DP with DBO/ubatching"
            )

        try:
            capacities = self.engine.ring_capacities()
            self._byte_capacity = capacities.effective_bytes
            self._task_capacity = capacities.task_entries
            self._max_num_batched_tokens = int(
                scheduler.max_num_batched_tokens
            )
            self._max_num_seqs = int(scheduler.max_num_seqs)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "DMI vLLM could not cache finite scheduler/ring bounds"
            ) from exc
        if (
            self._byte_capacity <= 0
            or self._task_capacity <= 0
            or self._max_num_batched_tokens <= 0
            or self._max_num_seqs <= 0
        ):
            raise RuntimeError("DMI vLLM scheduler/ring bounds must be positive")

        compilation = self.vllm_config.compilation_config
        capture_sizes = compilation.cudagraph_capture_sizes or ()
        try:
            self._capture_sizes = tuple(
                sorted({int(size) for size in capture_sizes if int(size) > 0})
            )
            self._max_capture_size = int(
                compilation.max_cudagraph_capture_size
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "DMI vLLM could not cache CUDA-graph capture bounds"
            ) from exc
        if self._capture_sizes and (
            self._max_capture_size <= 0
            or self._capture_sizes[-1] > self._max_capture_size
        ):
            raise RuntimeError("Invalid vLLM CUDA-graph capture sizes")

        tp_role_count = min(tp_size, 2)

        def add_term(
            terms: dict[int, int], row_bytes: int, multiplicity: int
        ) -> None:
            terms[row_bytes] = terms.get(row_bytes, 0) + multiplicity

        def compile_formula(
            specs: Tuple[HookSpec, ...]
        ) -> _VLLMRoleFormula:
            real_terms: dict[int, int] = {}
            execution_terms: dict[int, int] = {}
            request_terms: dict[int, int] = {}
            hook_count = 0
            for spec in specs:
                row_bytes = _row_bytes_for_spec(spec, cfg)
                if row_bytes <= 0:
                    continue
                row_basis = hook_row_basis(spec.hook_type)
                if row_basis is HookRowBasis.REQUEST_ROWS:
                    add_term(request_terms, row_bytes, 1)
                elif self.gpu_padding_strip and spec.dim0_is_actual_tokens:
                    add_term(real_terms, row_bytes, 1)
                else:
                    add_term(execution_terms, row_bytes, 1)
                hook_count += 1
            return _VLLMRoleFormula(
                real_terms=tuple(sorted(real_terms.items())),
                execution_terms=tuple(sorted(execution_terms.items())),
                request_terms=tuple(sorted(request_terms.items())),
                hook_count=hook_count,
            )

        candidate_formulas = tuple(
            compile_formula(specs)
            for specs in selection.candidate_rank_hook_sets
        )

        tp_rank, _dp_rank, _ep_rank, pp_rank = self.detect_parallel_ranks()
        local_tp_role = 0 if tp_rank == 0 else 1
        local_candidate_index = pp_rank * tp_role_count + local_tp_role
        if not 0 <= local_candidate_index < len(candidate_formulas):
            raise RuntimeError("DMI vLLM could not identify its local role")
        local_formula = candidate_formulas[local_candidate_index]
        concrete_formula = compile_formula(selection.local_hooks)
        if local_formula != concrete_formula:
            raise RuntimeError(
                "DMI vLLM cached role formula does not match local hook layout: "
                f"cached={local_formula}, concrete={concrete_formula}"
            )

        self._role_formulas = tuple(dict.fromkeys(candidate_formulas))
        self._local_role_formula = local_formula
        self._has_global_hooks = any(
            formula.hook_count > 0 for formula in self._role_formulas
        )
        self._captures_final_logits = any(
            spec.hook_type == HOOK_TYPE_FINAL_LOGITS
            for specs in selection.candidate_rank_hook_sets
            for spec in specs
        )
        max_hooks = max(
            (formula.hook_count for formula in self._role_formulas),
            default=0,
        )
        if max_hooks > self._task_capacity:
            raise RuntimeError(
                "DMI vLLM selected hooks exceed task-ring capacity: "
                f"hooks={max_hooks}, capacity={self._task_capacity}"
            )

    def _record_real_layout(
        self,
        scheduler_output: Any,
        model_runner: Any,
        num_scheduled_tokens: Any,
    ) -> None:
        """Copy the packed order produced by V1 input preparation."""
        input_batch = model_runner.input_batch
        num_reqs = int(input_batch.num_reqs)
        self._record_packed_layout(
            scheduler_output,
            num_reqs=num_reqs,
            raw_req_ids=input_batch.req_ids[:num_reqs],
            scheduled_counts=num_scheduled_tokens[:num_reqs],
            computed_counts=input_batch.num_computed_tokens_cpu[:num_reqs],
            boundary="_prepare_inputs",
        )

    def _record_v2_real_layout(
        self,
        scheduler_output: Any,
        input_batch: Any,
    ) -> None:
        """Copy the packed order produced by V2 ``prepare_inputs``."""
        num_reqs = int(input_batch.num_reqs)
        self._record_packed_layout(
            scheduler_output,
            num_reqs=num_reqs,
            raw_req_ids=input_batch.req_ids[:num_reqs],
            scheduled_counts=input_batch.num_scheduled_tokens[:num_reqs],
            computed_counts=input_batch.num_computed_tokens_np[:num_reqs],
            boundary="prepare_inputs",
        )

    def _record_packed_layout(
        self,
        scheduler_output: Any,
        *,
        num_reqs: int,
        raw_req_ids: Any,
        scheduled_counts: Any,
        computed_counts: Any,
        boundary: str,
    ) -> None:
        """Validate and snapshot one runner's CPU-side packed layout."""
        state = self._step_state
        if state.phase is not VLLMStepPhase.ARMED:
            raise RuntimeError(
                f"DMI vLLM observed duplicate or unarmed {boundary}"
            )
        if state.scheduler_output is not scheduler_output:
            raise RuntimeError("DMI vLLM scheduler output changed during prep")

        raw_req_ids = tuple(raw_req_ids)
        scheduled_counts = tuple(int(value) for value in scheduled_counts)
        computed_counts = tuple(int(value) for value in computed_counts)
        if (
            len(raw_req_ids) != num_reqs
            or len(scheduled_counts) != num_reqs
            or len(computed_counts) != num_reqs
        ):
            raise RuntimeError("DMI vLLM packed layout lengths disagree")
        if any(not isinstance(req_id, str) for req_id in raw_req_ids):
            raise RuntimeError("DMI vLLM request IDs must be strings")
        if len(set(raw_req_ids)) != num_reqs:
            raise RuntimeError("DMI vLLM packed request IDs are not unique")

        req_ids = tuple(
            normalize_vllm_request_id(req_id) for req_id in raw_req_ids
        )
        if len(set(req_ids)) != num_reqs:
            raise RuntimeError(
                "DMI vLLM normalized request IDs are not unique"
            )
        if any(value < 0 for value in scheduled_counts):
            raise RuntimeError("DMI vLLM scheduled token count is negative")
        if any(value < 0 for value in computed_counts):
            raise RuntimeError("DMI vLLM computed token count is negative")

        scheduler_counts = scheduler_output.num_scheduled_tokens
        if (
            len(scheduler_counts) != num_reqs
            or set(scheduler_counts) != set(raw_req_ids)
        ):
            raise RuntimeError(
                "DMI vLLM packed IDs do not match scheduler membership"
            )
        for req_id, count in zip(raw_req_ids, scheduled_counts):
            if int(scheduler_counts[req_id]) != count:
                raise RuntimeError(
                    f"DMI vLLM scheduled count mismatch for {req_id!r}"
                )

        total_rows = sum(scheduled_counts)
        if total_rows != int(scheduler_output.total_num_scheduled_tokens):
            raise RuntimeError(
                "DMI vLLM packed rows do not match scheduler total"
            )

        offsets: List[int] = []
        token_ranges: List[Tuple[int, int]] = []
        offset = 0
        for scheduled, computed in zip(
            scheduled_counts, computed_counts
        ):
            offsets.append(offset)
            token_ranges.append((computed, computed + scheduled))
            offset += scheduled
        if offset != total_rows:
            raise RuntimeError("DMI vLLM packed offsets do not cover all rows")

        state.layout = _VLLMRealLayout(
            raw_req_ids=raw_req_ids,
            req_ids=req_ids,
            scheduled_counts=scheduled_counts,
            computed_counts=computed_counts,
            token_ranges=tuple(token_ranges),
            dim0_offsets=tuple(offsets),
            total_rows=total_rows,
        )
        state.phase = VLLMStepPhase.LAYOUT_READY

    def _preview_exact_dispatch(
        self,
        model_runner: Any,
        *,
        num_tokens: int,
        num_reqs: int,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        force_eager: bool,
        force_uniform_decode: Optional[bool],
        force_has_lora: Optional[bool],
        force_num_active_loras: Optional[int],
        num_encoder_reqs: int,
    ) -> Tuple[Any, Any]:
        """Run only vLLM's read-only dispatcher calculation."""
        from vllm.config import CUDAGraphMode

        uniform_decode = model_runner._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=model_runner.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        has_encoder_output = (
            model_runner.model_config.is_encoder_decoder
            and num_encoder_reqs > 0
        )
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(model_runner.input_batch.lora_id_to_lora_request)
        )
        has_lora = (
            num_active_loras > 0
            if force_has_lora is None
            else force_has_lora
        )
        valid_modes = (
            {CUDAGraphMode.NONE}
            if force_eager
            else set(CUDAGraphMode.valid_runtime_modes())
        )
        invalid_modes = (
            {CUDAGraphMode.FULL}
            if use_cascade_attn or has_encoder_output
            else set()
        )
        return model_runner.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens,
            has_lora=has_lora,
            uniform_decode=uniform_decode,
            num_active_loras=num_active_loras,
            valid_modes=valid_modes,
            invalid_modes=invalid_modes,
        )

    def _conservative_execution_bound(
        self, num_tokens: int, num_reqs: int
    ) -> Tuple[int, int, int]:
        """Return bounded execution, real-row, and request-row counts."""
        parallel = self.vllm_config.parallel_config
        compilation = self.vllm_config.compilation_config
        if parallel.data_parallel_size > 1:
            real_rows = self._max_num_batched_tokens
            request_rows = self._max_num_seqs
        else:
            real_rows = int(num_tokens)
            request_rows = int(num_reqs)

        dispatch_rows = real_rows
        if compilation.pass_config.enable_sp:
            tp_size = int(parallel.tensor_parallel_size)
            dispatch_rows = (
                (dispatch_rows + tp_size - 1) // tp_size
            ) * tp_size

        execution_rows = dispatch_rows
        if (
            self._capture_sizes
            and dispatch_rows <= self._max_capture_size
        ):
            execution_rows = self._max_capture_size
        if execution_rows < dispatch_rows:
            raise RuntimeError(
                "DMI vLLM conservative graph bound is not an upper bound"
            )
        return execution_rows, real_rows, request_rows

    def _preflight_force_eager(
        self,
        model_runner: Any,
        *,
        num_tokens: int,
        num_reqs: int,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        caller_force_eager: bool,
        force_uniform_decode: Optional[bool],
        force_has_lora: Optional[bool],
        force_num_active_loras: Optional[int],
        num_encoder_reqs: int,
    ) -> bool:
        """Evaluate cached worst-role formulas with no transport effects."""
        state = self._step_state
        parallel = self.vllm_config.parallel_config
        compilation = self.vllm_config.compilation_config
        exact = (
            not compilation.pass_config.enable_sp
            and parallel.data_parallel_size == 1
            and not parallel.enable_expert_parallel
        )

        if exact:
            capacity_candidate = self._preview_exact_dispatch(
                model_runner,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=use_cascade_attn,
                force_eager=(
                    caller_force_eager or state.force_eager_latch
                ),
                force_uniform_decode=force_uniform_decode,
                force_has_lora=force_has_lora,
                force_num_active_loras=force_num_active_loras,
                num_encoder_reqs=num_encoder_reqs,
            )
            state.capacity_candidate = capacity_candidate
            state.execution_bound = None
            state.real_rows_bound = None
            state.request_rows_bound = None
            execution_rows = int(capacity_candidate[1].num_tokens)
            real_rows = int(num_tokens)
            request_rows = int(num_reqs)
        else:
            (
                execution_rows,
                real_rows,
                request_rows,
            ) = self._conservative_execution_bound(
                num_tokens, num_reqs
            )
            state.capacity_candidate = None
            state.execution_bound = execution_rows
            state.real_rows_bound = real_rows
            state.request_rows_bound = request_rows

        worst_bytes = max(
            (
                formula.bytes_for(
                    execution_rows, real_rows, request_rows
                )
                for formula in self._role_formulas
            ),
            default=0,
        )
        force_eager = worst_bytes > self._byte_capacity

        if (
            exact
            and self._validation_mode is VLLMValidationMode.VERIFY
        ):
            state.expected_dispatch = self._preview_exact_dispatch(
                model_runner,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=use_cascade_attn,
                force_eager=(
                    caller_force_eager
                    or state.force_eager_latch
                    or force_eager
                ),
                force_uniform_decode=force_uniform_decode,
                force_has_lora=force_has_lora,
                force_num_active_loras=force_num_active_loras,
                num_encoder_reqs=num_encoder_reqs,
            )
        else:
            state.expected_dispatch = None
        return force_eager

    def _preflight_v2_force_eager(
        self,
        batch_descriptor: Any,
        *,
        num_tokens: int,
        num_reqs: int,
    ) -> bool:
        """Check a V2 dispatch candidate before it becomes the real batch."""
        execution_rows = int(batch_descriptor.num_tokens)
        real_rows = int(num_tokens)
        request_rows = int(num_reqs)
        if execution_rows < real_rows:
            raise RuntimeError(
                "DMI vLLM V2 dispatch has fewer rows than scheduled tokens"
            )

        state = self._step_state
        state.capacity_candidate = (
            batch_descriptor.cg_mode,
            batch_descriptor,
        )
        state.execution_bound = None
        state.real_rows_bound = None
        state.request_rows_bound = None
        worst_bytes = max(
            (
                formula.bytes_for(
                    execution_rows,
                    real_rows,
                    request_rows,
                )
                for formula in self._role_formulas
            ),
            default=0,
        )
        return worst_bytes > self._byte_capacity

    def _commit_actual_dispatch(
        self,
        scheduler_output: Any,
        model_runner: Any,
        cudagraph_mode: Any,
        batch_descriptor: Any,
        combined_force_eager: bool,
    ) -> None:
        """Commit metadata/reservation from the real layout and descriptor."""
        state = self._step_state
        layout = state.layout
        if state.phase is not VLLMStepPhase.LAYOUT_READY or layout is None:
            raise RuntimeError(
                "DMI vLLM dispatch committed without one real layout"
            )
        if int(batch_descriptor.num_tokens) < layout.total_rows:
            raise RuntimeError(
                "DMI vLLM dispatch has fewer rows than the packed layout"
            )

        state.real_cudagraph_mode = cudagraph_mode
        state.real_batch_descriptor = batch_descriptor

        if self._validation_mode is VLLMValidationMode.VERIFY:
            if state.expected_dispatch is not None:
                if state.expected_dispatch != (
                    cudagraph_mode,
                    batch_descriptor,
                ):
                    raise RuntimeError(
                        "DMI vLLM exact dispatch preview disagrees with vLLM"
                    )
            elif state.execution_bound is not None:
                if int(batch_descriptor.num_tokens) > state.execution_bound:
                    raise RuntimeError(
                        "DMI vLLM execution rows exceed conservative bound"
                    )
                if (
                    state.real_rows_bound is None
                    or layout.total_rows > state.real_rows_bound
                    or state.request_rows_bound is None
                    or len(layout.req_ids) > state.request_rows_bound
                ):
                    raise RuntimeError(
                        "DMI vLLM real layout exceeds conservative bound"
                    )

        ctx = self.build_step_context(scheduler_output, model_runner)
        if ctx is None:
            raise RuntimeError("DMI vLLM committed an empty armed step")
        actual_plan = self.plan_step(ctx)
        actual_bytes, actual_hooks, actual_needs_eager = actual_plan
        if actual_hooks > self._task_capacity:
            raise RuntimeError(
                "DMI vLLM actual hook count exceeds cached task capacity"
            )
        actual_requires_eager = (
            actual_needs_eager
            or actual_bytes > self._byte_capacity
        )
        if actual_requires_eager and not combined_force_eager:
            raise RuntimeError(
                "DMI vLLM eager preflight produced a false negative"
            )

        if self._validation_mode is VLLMValidationMode.VERIFY:
            formula = self._local_role_formula
            if formula is None:
                raise RuntimeError("DMI vLLM local role formula is missing")
            formula_plan = (
                formula.bytes_for(
                    int(batch_descriptor.num_tokens),
                    layout.total_rows,
                    len(layout.req_ids),
                ),
                formula.hook_count,
            )
            if formula_plan != actual_plan[:2] or actual_needs_eager:
                raise RuntimeError(
                    "DMI vLLM cached local formula disagrees with actual plan: "
                    f"cached={formula_plan}, actual={actual_plan}"
                )

        self.commit_step(ctx, actual_plan)

        if (self.gpu_padding_strip
                and self._pinned_row_count is not None
                and self._row_count_dev is not None
                and self._last_total_q > 0):
            # CPU: write the current step's actual_q_len to pinned host.
            # GPU: enqueue an async copy to the shared device scalar on
            # the model stream (captureable; values propagate to replays).
            self._pinned_row_count[0] = self._last_total_q
            self._row_count_dev.copy_(self._pinned_row_count, non_blocking=True)
        self._last_committed_layout = layout
        state.phase = VLLMStepPhase.COMMITTED


def _row_bytes_for_spec(spec, model_cfg) -> int:
    """Bytes-per-token for `spec`'s shape under vLLM flat layout.

    Computed as prod(shape_excluding_total_tokens) * elem_size where
    the shape comes from _compute_hook_shape with batch=0, q_len=1.
    The "1" stands in for "one token's worth"; the result is the
    per-token byte stride.  Returns 0 if the spec produces an empty
    shape (skip).
    """
    shape = compute_hook_shape(
        spec.hook_type, model_cfg, batch=0, q_len=1, kv_dim=0,
        logits_to_keep=0,
    )
    if not shape:
        return 0
    # In flat mode, dim-0 of the spec's shape IS the token count.
    # shape[0] should be 1 here (since we passed q_len=1); the
    # remaining dims times elem_size = bytes per token.
    nelem = 1
    for d in shape[1:]:
        nelem *= d
    dtype = spec.dtype if spec.dtype is not None else model_cfg.dtype
    elem_size = torch._utils._element_size(dtype)
    return int(nelem) * int(elem_size)


# ---------------------------------------------------------------------------
# DMXGPUWorker
# ---------------------------------------------------------------------------


class DMXV2GPUWorker(Worker):
    """vLLM ``Worker`` subclass that owns a ``VLLMAdaptor`` and
    delegates per-step work to it.

    Lifecycle:
      ``init_device``           -- super + build engine + adaptor + null mode +
                                    wrap input preparation and dispatch.
      ``load_model``            -- arch remap + super + adaptor.attach_model.
      ``compile_or_warm_up_model`` -- super + clear null_mode (unless user
                                       opted-in via ``dmx_null_mode``).
      ``execute_model``         -- arm per-call state + super; the dispatch
                                    wrapper commits DMI metadata before forward.
      ``stop_monitoring``       -- adaptor.close (CUDA sync, ring stop,
                                    deactivate transport, host engine stop).
      ``shutdown``              -- best-effort stop_monitoring + super.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.adaptor: Optional[VLLMAdaptor] = None
        self._dmx_host_engine: Any = None
        self._dmx_requested_hook_selection: str = "vllm-full"
        self._dmx_hook_selection: str = "vllm-full"
        self._dmx_stopped = False
        self._dmx_v2_dispatch_manager: Any = None
        self._dmx_v2_original_dispatch: Any = None

    def _validate_dmi_config(self) -> None:
        """Reject unsupported runner configurations before CUDA init."""
        if not getattr(self, "use_v2_model_runner", False):
            raise RuntimeError(
                "DMXV2GPUWorker requires "
                "vLLM's V2 model runner"
            )

        config = self.vllm_config
        model = config.model_config
        parallel = config.parallel_config
        additional_config = getattr(config, "additional_config", None)
        if not isinstance(additional_config, dict):
            additional_config = {}
        hook_selection = _cfg(
            additional_config,
            "dmx_hook_selection",
            "DMX_HOOK_SELECTION",
            "vllm-full",
        )
        try:
            selected_hook_types, resolved_selection_tokens = (
                _resolve_hook_selection_tokens(hook_selection)
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"DMI vLLM received an invalid hook selection: {hook_selection!r}"
            ) from exc
        self._dmx_requested_hook_selection = hook_selection
        effective_hook_selection = hook_selection
        selection_warning = None
        model_impl = getattr(model, "model_impl", "auto")
        if model_impl not in {"auto", "vllm"}:
            raise RuntimeError(
                "DMI vLLM supports only the auto or vllm model "
                "implementation"
            )
        architectures = require_supported_architecture(model)
        validate_model_specific_config(config, architectures)
        if model.runner_type != "generate":
            raise RuntimeError("DMI vLLM supports only the generate runner")
        if int(parallel.data_parallel_size) > 1:
            raise RuntimeError("DMI vLLM does not support data parallelism")
        if int(getattr(parallel, "prefill_context_parallel_size", 1)) > 1:
            raise RuntimeError(
                "DMI vLLM does not support prefill context parallelism"
            )
        if parallel.use_ubatching:
            raise RuntimeError("DMI vLLM does not support DBO/ubatching")
        if getattr(parallel, "enable_elastic_ep", False):
            raise RuntimeError(
                "DMI vLLM does not support elastic expert parallelism"
            )
        if (
            getattr(self, "use_v2_model_runner", False)
            and config.speculative_config is not None
        ):
            raise RuntimeError(
                "DMI vLLM V2 model-runner support does not yet include "
                "speculative decoding"
            )
        if (
            getattr(parallel, "enable_batch_sharded_sampling", False)
            and HOOK_TYPE_FINAL_LOGITS in selected_hook_types
        ):
            raise RuntimeError(
                "DMI final_logits capture does not support batch-sharded "
                "sampling: compute_logits_local has sharded vocabulary and "
                "request rows. Disable enable_batch_sharded_sampling or select "
                "hooks without final_logits. Hidden-state capture is unaffected."
            )
        if HOOK_TYPE_TOKEN_IDS in selected_hook_types:
            unavailable_reason = _token_ids_unavailability_reason(model)
            if unavailable_reason is not None:
                explicit_token_selectors = tuple(
                    token
                    for token, hook_types in resolved_selection_tokens
                    if hook_types == frozenset({HOOK_TYPE_TOKEN_IDS})
                )
                if explicit_token_selectors:
                    selectors = ", ".join(
                        repr(token) for token in explicit_token_selectors
                    )
                    raise RuntimeError(
                        "DMI token-ID capture was explicitly selected by "
                        f"{selectors}, but input_ids cannot reach the model: "
                        f"{unavailable_reason}. Select hooks without token_ids."
                    )
                selected_hook_types = frozenset(
                    hook_type
                    for hook_type in selected_hook_types
                    if hook_type != HOOK_TYPE_TOKEN_IDS
                )
                effective_hook_selection = _effective_selection_preset(
                    selected_hook_types
                )
                selection_warning = (
                    "[vllm_integration] Removed token_ids from indirect hook "
                    f"selection {hook_selection!r}: {unavailable_reason}. "
                    f"Effective selection is {effective_hook_selection!r} "
                    "(the requested hook set minus token_ids)."
                )
        if (
            config.speculative_config is not None
            and HOOK_TYPE_FINAL_LOGITS in selected_hook_types
        ):
            raise RuntimeError(
                "DMI final-logit capture does not support speculative decoding"
            )
        self._dmx_hook_selection = effective_hook_selection
        if selection_warning is not None:
            warnings.warn(selection_warning, UserWarning, stacklevel=2)

    def _install_v2_prepare_wrapper(self) -> None:
        """Commit V2's prepared layout before attention and model forward."""
        adaptor = self.adaptor
        if adaptor is None:
            raise RuntimeError("DMI vLLM V2 wrapper requires an adaptor")
        model_runner = self.model_runner
        original_prepare = model_runner.prepare_inputs

        def _wrapped_prepare_inputs(
            scheduler_output: Any,
            batch_req_state: Any,
            batch_desc: Any,
        ) -> Any:
            input_batch = original_prepare(
                scheduler_output, batch_req_state, batch_desc
            )
            state = adaptor._step_state
            if state.phase is VLLMStepPhase.ARMED:
                adaptor._record_v2_real_layout(
                    scheduler_output,
                    input_batch,
                )
                if state.capacity_candidate is None:
                    force_eager = adaptor._preflight_v2_force_eager(
                        batch_desc,
                        num_tokens=input_batch.num_tokens,
                        num_reqs=input_batch.num_reqs,
                    )
                    if force_eager:
                        from vllm.config import CUDAGraphMode

                        if batch_desc.cg_mode is not CUDAGraphMode.NONE:
                            raise RuntimeError(
                                "DMI vLLM V2 dispatch bypassed the "
                                "eager-capacity preflight"
                            )
                        # Encoder/model-specific paths can bypass the graph
                        # manager after vLLM has selected eager execution.
                        state.force_eager_latch = True
                if adaptor._validation_mode is VLLMValidationMode.VERIFY:
                    state.expected_dispatch = (
                        batch_desc.cg_mode,
                        batch_desc,
                    )
                adaptor._commit_actual_dispatch(
                    scheduler_output,
                    model_runner,
                    batch_desc.cg_mode,
                    batch_desc,
                    state.force_eager_latch,
                )
            elif state.phase is not VLLMStepPhase.IDLE:
                raise RuntimeError(
                    "DMI vLLM observed a second V2 prepare_inputs in one step"
                )
            return input_batch

        model_runner.prepare_inputs = _wrapped_prepare_inputs

    def _ensure_v2_dispatch_wrapper(self) -> None:
        """Wrap V2 graph selection once its manager has been initialized."""
        manager = getattr(self.model_runner, "cudagraph_manager", None)
        if manager is None:
            raise RuntimeError(
                "DMI vLLM V2 runner has no initialized CUDA-graph manager"
            )
        if getattr(self, "_dmx_v2_dispatch_manager", None) is manager:
            return
        if getattr(self, "_dmx_v2_dispatch_manager", None) is not None:
            raise RuntimeError("DMI vLLM V2 CUDA-graph manager changed at runtime")

        adaptor = self.adaptor
        if adaptor is None:
            raise RuntimeError("DMI vLLM V2 dispatch wrapper requires an adaptor")
        original_dispatch = manager.dispatch

        def _wrapped_dispatch(
            num_reqs: int,
            num_tokens: int,
            uniform_token_count: Optional[int],
            num_active_loras: int,
            max_query_len: Optional[int] = None,
        ) -> Any:
            candidate = original_dispatch(
                num_reqs,
                num_tokens,
                uniform_token_count,
                num_active_loras,
                max_query_len=max_query_len,
            )
            state = adaptor._step_state
            if state.phase is VLLMStepPhase.IDLE:
                return candidate
            if state.phase is not VLLMStepPhase.ARMED:
                raise RuntimeError(
                    f"DMI vLLM observed V2 dispatch in phase {state.phase}"
                )

            force_eager = adaptor._preflight_v2_force_eager(
                candidate,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
            )
            state.force_eager_latch |= force_eager
            if force_eager:
                from vllm.config import CUDAGraphMode
                from vllm.v1.worker.gpu.cudagraph_utils import (
                    BatchExecutionDescriptor,
                )

                candidate = BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_active_loras=num_active_loras,
                )
            if adaptor._validation_mode is VLLMValidationMode.VERIFY:
                state.expected_dispatch = (
                    candidate.cg_mode,
                    candidate,
                )
            return candidate

        manager.dispatch = _wrapped_dispatch
        self._dmx_v2_dispatch_manager = manager
        self._dmx_v2_original_dispatch = original_dispatch

    def _restore_v2_dispatch_wrapper(self) -> None:
        manager = getattr(self, "_dmx_v2_dispatch_manager", None)
        original_dispatch = getattr(self, "_dmx_v2_original_dispatch", None)
        if manager is not None and original_dispatch is not None:
            manager.dispatch = original_dispatch
        self._dmx_v2_dispatch_manager = None
        self._dmx_v2_original_dispatch = None

    def init_device(self) -> None:
        self._validate_dmi_config()
        super().init_device()

        ac = self.vllm_config.additional_config
        if not isinstance(ac, dict):
            ac = {}

        model_id = _cfg(ac, "dmx_model_id", "DMX_MODEL_ID", "")
        ring_payload_mb = _cfg(ac, "dmx_ring_payload_mb", "DMX_RING_PAYLOAD_MB", 4096)
        ring_pinned_mb = _cfg(ac, "dmx_ring_pinned_mb", "DMX_RING_PINNED_MB", 4096)
        ring_entries = _cfg(ac, "dmx_ring_task_entries", "DMX_RING_TASK_ENTRIES", 65536)
        null_mode = _cfg(ac, "dmx_null_mode", "DMX_NULL_MODE", False)
        db_host = _cfg(ac, "dmx_db_host", "DMX_DB_HOST", "")
        db_port = _cfg(ac, "dmx_db_port", "DMX_DB_PORT", 9000)
        db_database = _cfg(ac, "dmx_db_database", "DMX_DB_DATABASE", "default")
        db_table = _cfg(ac, "dmx_db_table", "DMX_DB_TABLE", "offload")
        ch_parallelism = int(
            _cfg(ac, "dmx_ch_parallelism", "DMX_CH_PARALLELISM", 10)
        )

        resolved_model_id = model_id or str(self.vllm_config.model_config.model)

        # Host engine (ClickHouse), optional.
        host_engine = None
        if db_host:
            ch_cfg = ClickHouseClientConfig()
            ch_cfg.host = db_host
            ch_cfg.port = db_port
            ch_cfg.database = db_database
            ch_cfg.table = db_table
            ch_cfg.create_database_if_missing = True
            stage_cfg = StageConfig.clickhouse_insert(
                ch_cfg, parallelism=ch_parallelism, name="clickhouse_insert"
            )
            q = stage_cfg.input_queue
            q.max_batch_items = int(_cfg(
                ac, "dmx_ch_max_batch_items", "DMX_CH_MAX_BATCH_ITEMS", 1024))
            q.high_watermark_items = q.max_batch_items
            q.max_batch_size = int(_cfg(
                ac, "dmx_ch_max_batch_bytes", "DMX_CH_MAX_BATCH_BYTES",
                2048 * 1024 * 1024))
            q.high_watermark_size = q.max_batch_size
            host_engine = DMXHostEngine(stage_cfg)
            self._dmx_host_engine = host_engine

        ring_cfg = RingConfig()
        ring_cfg.payload_ring_bytes = ring_payload_mb * 1024 * 1024
        ring_cfg.pinned_staging_bytes = ring_pinned_mb * 1024 * 1024
        ring_cfg.task_ring_entries = ring_entries
        ring_cfg.drain_flush_timeout_us = int(_cfg(
            ac, "dmx_drain_flush_timeout_us", "DMX_DRAIN_FLUSH_TIMEOUT_US",
            0))

        # MonitoringEngine + ring transport.
        engine = MonitoringEngine(
            config=None,
            model_id=resolved_model_id,
            host_engine=host_engine,
            ring_config=ring_cfg,
        )

        # Build the adaptor.  Hooks aren't installed yet -- that
        # happens in load_model after the model is materialized.
        # gpu_padding_strip is on by default; flip it off via
        # additional_config["dmx_gpu_padding_strip"]=False or
        # DMX_GPU_PADDING_STRIP=0 if needed for debugging.
        gpu_padding_strip = _cfg(
            ac, "dmx_gpu_padding_strip", "DMX_GPU_PADDING_STRIP", True)
        self.adaptor = VLLMAdaptor(
            engine, resolved_model_id, self.vllm_config,
            gpu_padding_strip=bool(gpu_padding_strip))
        self.adaptor.user_wants_null_mode = bool(null_mode)

        # Enable null mode for warmup so producer kernels fire (CUDA
        # graph capture needs them) but no-op on the data path.  Cleared
        # after warmup in compile_or_warm_up_model unless the user
        # explicitly opted in to permanent null mode.
        engine.set_capture_enabled(False)

        # DMI's transport and native active-engine slots are process/device
        # global. This integration therefore assumes one active monitored
        # engine/forward owner per process.
        if getattr(self, "use_v2_model_runner", False):
            self._install_v2_prepare_wrapper()

            # Backwards-compat attributes for external subclasses that read
            # the pre-refactor names (e.g. tests/compare_worker.py reads
            # `self._dmx_tp_rank` + `self._dmx_tp_size` to format per-rank
            # filenames).
            from vllm.distributed.parallel_state import get_tp_group
            tp_rank, dp_rank, ep_rank, pp_rank = (
                self.adaptor.detect_parallel_ranks()
            )
            self._dmx_tp_rank = tp_rank
            self._dmx_dp_rank = dp_rank
            self._dmx_ep_rank = ep_rank
            self._dmx_pp_rank = pp_rank
            self._dmx_tp_size = get_tp_group().world_size
            return

        adaptor = self.adaptor
        model_runner = self.model_runner
        orig_prepare = model_runner._prepare_inputs
        orig_fn = self.model_runner._determine_batch_execution_and_padding

        def _wrapped_prepare(
            scheduler_output: Any,
            num_scheduled_tokens: np.ndarray,
        ) -> Any:
            result = orig_prepare(
                scheduler_output,
                num_scheduled_tokens,
            )
            phase = adaptor._step_state.phase
            if phase is VLLMStepPhase.ARMED:
                adaptor._record_real_layout(
                    scheduler_output,
                    model_runner,
                    num_scheduled_tokens,
                )
            elif phase is not VLLMStepPhase.IDLE:
                raise RuntimeError(
                    "DMI vLLM observed a second _prepare_inputs in one step"
                )
            return result

        def _wrapped_determine(
            num_tokens: int,
            num_reqs: int,
            num_scheduled_tokens_np: np.ndarray,
            max_num_scheduled_tokens: int,
            use_cascade_attn: bool,
            allow_microbatching: bool = True,
            force_eager: bool = False,
            force_uniform_decode: Optional[bool] = None,
            force_has_lora: Optional[bool] = None,
            force_num_active_loras: Optional[int] = None,
            num_encoder_reqs: int = 0,
        ) -> Any:
            state = adaptor._step_state
            if state.phase is VLLMStepPhase.IDLE:
                return orig_fn(
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    use_cascade_attn=use_cascade_attn,
                    allow_microbatching=allow_microbatching,
                    force_eager=force_eager,
                    force_uniform_decode=force_uniform_decode,
                    force_has_lora=force_has_lora,
                    force_num_active_loras=force_num_active_loras,
                    num_encoder_reqs=num_encoder_reqs,
                )
            if state.phase is VLLMStepPhase.COMMITTED:
                raise RuntimeError(
                    "DMI vLLM observed dispatch after metadata commit"
                )
            if state.phase not in {
                VLLMStepPhase.ARMED,
                VLLMStepPhase.LAYOUT_READY,
            }:
                raise RuntimeError(
                    f"DMI vLLM invalid dispatch phase: {state.phase}"
                )

            state.force_eager_latch |= adaptor._preflight_force_eager(
                model_runner,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=use_cascade_attn,
                caller_force_eager=force_eager,
                force_uniform_decode=force_uniform_decode,
                force_has_lora=force_has_lora,
                force_num_active_loras=force_num_active_loras,
                num_encoder_reqs=num_encoder_reqs,
            )
            combined_force_eager = force_eager or state.force_eager_latch
            result = orig_fn(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=use_cascade_attn,
                allow_microbatching=allow_microbatching,
                force_eager=combined_force_eager,
                force_uniform_decode=force_uniform_decode,
                force_has_lora=force_has_lora,
                force_num_active_loras=force_num_active_loras,
                num_encoder_reqs=num_encoder_reqs,
            )

            if state.phase is VLLMStepPhase.ARMED:
                parallel = adaptor.vllm_config.parallel_config
                compilation = adaptor.vllm_config.compilation_config
                if (
                    parallel.pipeline_parallel_size <= 1
                    or not compilation.pass_config.enable_sp
                    or state.prelayout_dispatch_seen
                ):
                    raise RuntimeError(
                        "DMI vLLM observed unexpected pre-layout dispatch"
                    )
                state.prelayout_dispatch_seen = True
                if (
                    adaptor._validation_mode
                    is VLLMValidationMode.VERIFY
                    and state.execution_bound is not None
                    and int(result[1].num_tokens) > state.execution_bound
                ):
                    raise RuntimeError(
                        "DMI vLLM early dispatch exceeds conservative bound"
                    )
                return result

            adaptor._commit_actual_dispatch(
                state.scheduler_output,
                model_runner,
                result[0],
                result[1],
                combined_force_eager,
            )
            return result

        model_runner._prepare_inputs = _wrapped_prepare
        model_runner._determine_batch_execution_and_padding = _wrapped_determine

        # Backwards-compat attributes for external subclasses that read
        # the pre-refactor names (e.g. tests/compare_worker.py reads
        # `self._dmx_tp_rank` + `self._dmx_tp_size` to format per-rank
        # filenames).
        from vllm.distributed.parallel_state import get_tp_group
        tp_rank, dp_rank, ep_rank, pp_rank = self.adaptor.detect_parallel_ranks()
        self._dmx_tp_rank = tp_rank
        self._dmx_dp_rank = dp_rank
        self._dmx_ep_rank = ep_rank
        self._dmx_pp_rank = pp_rank
        self._dmx_tp_size = get_tp_group().world_size

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        # Remap the architecture string in the HF config so vLLM's
        # registry resolves to the hooked variant (Qwen3PForCausalLM
        # etc.).  Mutates vllm_config in place.
        hf_cfg = self.vllm_config.model_config.hf_config
        archs = getattr(hf_cfg, "architectures", [])
        new_archs = [_ARCH_REMAP.get(a, a) for a in archs]
        hf_cfg.architectures = new_archs

        super().load_model(load_dummy_weights=load_dummy_weights)

        # Now the model is materialized; install hooks via the adapter.
        if self.adaptor is None:
            return
        self.adaptor.attach_model(
            self.model_runner.model, hook_selection=self._dmx_hook_selection
        )

    def compile_or_warm_up_model(self) -> Any:
        # Warmup runs with null_mode=True (set in init_device).
        # Producer kernels fire but are no-ops -- ring stays clean.
        result = super().compile_or_warm_up_model()

        # Warmup done.  Turn off null mode unless the user explicitly
        # asked for permanent null mode.  set_null_mode does
        # cudaDeviceSynchronize internally so it's safe to call here.
        if (
            self.adaptor is not None
            and not self.adaptor.user_wants_null_mode
        ):
            self.adaptor.engine.set_capture_enabled(True)

        return result

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        # vLLM can send a final zero-token scheduler update after a request
        # finishes.  That path only retires scheduler state and never invokes
        # the model, so it remains safe after terminal monitoring shutdown.
        # Reject every step that could execute hooked model code.
        if (
            getattr(self, "_dmx_stopped", False)
            and scheduler_output.total_num_scheduled_tokens > 0
        ):
            raise RuntimeError(
                "DMI monitoring has been stopped permanently for this worker"
            )
        adaptor = self.adaptor
        ec_non_consumer = (
            has_ec_transfer() and not get_ec_transfer().is_consumer
        )
        if (
            adaptor is None
            or scheduler_output.total_num_scheduled_tokens <= 0
            or not adaptor.engine.capture_enabled
            or not adaptor._has_global_hooks
            # EC non-consumers return after state update without preparing or
            # executing a model batch, so there is no DMI step to commit.
            or ec_non_consumer
        ):
            return super().execute_model(scheduler_output)
        captures_final_logits = getattr(
            adaptor, "_captures_final_logits", None
        )
        if captures_final_logits is None:
            captures_final_logits = any(
                spec.hook_type == HOOK_TYPE_FINAL_LOGITS
                for spec in adaptor.active_hook_specs
            )
        if captures_final_logits:
            active_prompt_logprobs = bool(
                getattr(
                    getattr(self.model_runner, "prompt_logprobs_worker", None),
                    "in_progress_prompt_logprobs",
                    None,
                )
            )
            new_prompt_logprobs = any(
                getattr(
                    getattr(request, "sampling_params", None),
                    "prompt_logprobs",
                    None,
                )
                is not None
                for request in getattr(
                    scheduler_output, "scheduled_new_reqs", ()
                )
            )
            if active_prompt_logprobs or new_prompt_logprobs:
                raise RuntimeError(
                    "DMI vLLM cannot capture final_logits for requests that "
                    "enable prompt_logprobs"
                )
        if adaptor._step_state.phase is not VLLMStepPhase.IDLE:
            raise RuntimeError("DMI vLLM execute_model calls overlap")

        adaptor._last_committed_layout = None
        adaptor._step_state = _VLLMStepState(
            phase=VLLMStepPhase.ARMED,
            scheduler_output=scheduler_output,
        )
        state = adaptor._step_state
        try:
            if getattr(self, "use_v2_model_runner", False):
                self._ensure_v2_dispatch_wrapper()
            result = super().execute_model(scheduler_output)
            if state.phase is not VLLMStepPhase.COMMITTED:
                raise RuntimeError(
                    "DMI vLLM forward completed without metadata commit"
                )
            return result
        finally:
            adaptor._step_state = _VLLMStepState()

    def execute_dummy_batch(self) -> None:
        if getattr(self, "_dmx_stopped", False):
            raise RuntimeError(
                "DMI monitoring has been stopped permanently for this worker"
            )
        super().execute_dummy_batch()

    def stop_monitoring(self) -> None:
        """Flush and stop DMI engine.  Reentrant: second call no-ops."""
        if getattr(self, "_dmx_stopped", False):
            return
        self._dmx_stopped = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._restore_v2_dispatch_wrapper()
        if self.adaptor is not None:
            adaptor = self.adaptor
            self.adaptor = None
            adaptor.close()

        if self._dmx_host_engine is not None:
            host_engine = self._dmx_host_engine
            self._dmx_host_engine = None
            host_engine.raise_if_failed()

    def shutdown(self) -> None:
        import logging
        if self.adaptor is not None:
            logging.getLogger(__name__).warning(
                "DMI engine not explicitly stopped before shutdown. "
                "Data may be incomplete. Call stop_monitoring() first."
            )
        # Best-effort flush (may be killed by vLLM's 8 s deadline).
        try:
            self.stop_monitoring()
        except Exception:
            logging.getLogger(__name__).exception(
                "DMI monitoring shutdown reported a backend failure"
            )
        super().shutdown()


def _attach_dmi_internal(outputs, model_id, reader=None, handles=None):
    """Tag final vLLM outputs with lazy DMI readback handles."""
    for r in outputs:
        # RequestOutput.request_id is already the external ID and must be used
        # verbatim. Parallel sampling creates scheduler children named
        # ``{output_index}_{parent_id}``, after random-suffix normalization.
        completions = getattr(r, "outputs", ())
        if len(completions) > 1:
            request_ids = [
                f"{completion.index}_{r.request_id}"
                for completion in completions
            ]
        else:
            request_ids = [r.request_id]
        handle = make_lazy_internal(
            model_id,
            reader=reader,
            request_ids=request_ids,
        )
        r.dmi_internal = handle
        if handles is not None:
            handles.add(handle)
    return outputs


class DMILLM(LLM):
    """Drop-in replacement for vLLM's ``LLM`` that captures model internals.

    Injects the DMI worker (so ``worker_cls`` need not be passed) and tags every
    final ``RequestOutput`` returned by ``generate``, ``chat``, or
    ``wait_for_completion`` with a lazy ``.dmi_internal``. Pass DMI settings
    through ``additional_config`` exactly as with plain ``LLM``
    (``dmx_model_id``, ``dmx_db_host``, ``dmx_db_port``, ``dmx_hook_selection``, ...).
    """

    _WORKER_CLS = "dmi_vllm_integration.v2.worker.DMXV2GPUWorker"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        worker_cls = kwargs.get("worker_cls")
        if worker_cls is not None and worker_cls not in (
            self._WORKER_CLS,
            DMXV2GPUWorker,
        ):
            raise ValueError(
                "DMILLM requires worker_cls="
                f"{self._WORKER_CLS!r}; got {worker_cls!r}"
            )
        kwargs["worker_cls"] = self._WORKER_CLS

        raw_ac = kwargs.get("additional_config")
        if raw_ac is None:
            ac: dict[str, Any] = {}
        elif isinstance(raw_ac, dict):
            ac = dict(raw_ac)
        else:
            raise TypeError("DMILLM additional_config must be a dict")

        default_model_id = f"dmi-vllm-{uuid4().hex}"
        self._dmx_model_id = str(
            _cfg(ac, "dmx_model_id", "DMX_MODEL_ID", "")
            or default_model_id
        )
        self._dmx_db_host = str(
            _cfg(ac, "dmx_db_host", "DMX_DB_HOST", "localhost")
        )
        if not self._dmx_db_host:
            raise ValueError(
                "DMILLM requires a nonempty dmx_db_host for persistence"
            )
        self._dmx_db_port = int(
            _cfg(ac, "dmx_db_port", "DMX_DB_PORT", 9000)
        )
        self._dmx_db_database = str(
            _cfg(ac, "dmx_db_database", "DMX_DB_DATABASE", "default")
        )
        self._dmx_db_table = str(
            _cfg(ac, "dmx_db_table", "DMX_DB_TABLE", "offload")
        )
        null_mode = bool(
            _cfg(ac, "dmx_null_mode", "DMX_NULL_MODE", False)
        )
        if null_mode:
            raise ValueError("DMILLM does not support dmx_null_mode")

        ac.update(
            dmx_model_id=self._dmx_model_id,
            dmx_db_host=self._dmx_db_host,
            dmx_db_port=self._dmx_db_port,
            dmx_db_database=self._dmx_db_database,
            dmx_db_table=self._dmx_db_table,
            dmx_null_mode=False,
        )
        kwargs["additional_config"] = ac
        self._dmx_reader: Optional[CHClickhouseDriverReadOnly] = None
        self._dmx_handles: WeakSet[Any] = WeakSet()
        self._dmx_stopped = False
        super().__init__(*args, **kwargs)

    def _ensure_dmi_active(self) -> None:
        if self._dmx_stopped:
            raise RuntimeError(
                "DMI monitoring has been stopped permanently for this engine"
            )

    def _add_request(self, *args: Any, **kwargs: Any) -> str:
        self._ensure_dmi_active()
        return super()._add_request(*args, **kwargs)

    def _run_engine(self, *args: Any, **kwargs: Any):
        self._ensure_dmi_active()
        outputs = super()._run_engine(*args, **kwargs)
        if self._dmx_reader is None:
            self._dmx_reader = CHClickhouseDriverReadOnly(
                host=self._dmx_db_host,
                port=self._dmx_db_port,
                database=self._dmx_db_database,
                table=self._dmx_db_table,
            )
        return _attach_dmi_internal(
            outputs,
            self._dmx_model_id,
            self._dmx_reader,
            self._dmx_handles,
        )

    def stop_monitoring(self) -> None:
        """Drain every worker, then close this wrapper's readback client."""
        if self._dmx_stopped:
            return
        self._dmx_stopped = True
        reader = self._dmx_reader
        try:
            self.collective_rpc("stop_monitoring")
            for handle in self._dmx_handles:
                handle.clear_cache()
        finally:
            if reader is not None:
                reader.close()
            self._dmx_reader = None


__all__ = [
    "VLLMAdaptor",
    "DMXV2GPUWorker",
    "DMILLM",
    "normalize_vllm_request_id",
]
