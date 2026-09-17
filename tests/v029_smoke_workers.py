"""Identical output-only logit tap on stock and monitored workers.

This diagnostic tap is separate from the API/storage oracle and does not replace
any model arithmetic. Each worker runs in its own engine subprocess.
"""

import os
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker
from dmi_vllm_integration.adapter import DMXGPUWorker
from dmi_vllm_integration.v2.worker import DMXV2GPUWorker


class _LogitTap:
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        self._smoke_active = False
        self._smoke_logits = []
        original = self.model_runner.model.compute_logits

        def tapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if self._smoke_active:
                self._smoke_logits.append(result.detach().cpu().clone())
            return result

        self.model_runner.model.compute_logits = tapped
        self._smoke_residuals = []
        self._smoke_pending_residuals = []
        self._smoke_computed = {}
        self._smoke_residual_enabled = bool(os.environ.get("DMI_SMOKE_RESIDUALS_PATH"))
        if self._smoke_residual_enabled:
            assert self.vllm_config.model_config.enforce_eager
            assert type(self.model_runner).__module__ == "vllm.v1.worker.gpu_model_runner"
            from tests.v029_residual_reference import old_residual_expression

            def observe(name, layer):
                def before_norm(_module, inputs):
                    if self._smoke_active:
                        self._smoke_pending_residuals.append(
                            (name, layer, old_residual_expression(inputs))
                        )
                return before_norm

            body = self.model_runner.model.model
            self._smoke_residual_family_count = 2 * len(body.layers) + 1
            for index, layer in enumerate(body.layers):
                layer.input_layernorm.register_forward_pre_hook(
                    observe("blocks.hook_resid_pre", index))
                layer.post_attention_layernorm.register_forward_pre_hook(
                    observe("blocks.hook_resid_mid", index))
            body.norm.register_forward_pre_hook(observe("hook_resid_final", -1))

    def execute_model(self, scheduler_output, *args, **kwargs):
        result = super().execute_model(scheduler_output, *args, **kwargs)
        if (self._smoke_residual_enabled and self._smoke_active
                and scheduler_output.total_num_scheduled_tokens > 0):
            # Independent stock V1 layout, not the DMI committed descriptor.
            batch = self.model_runner.input_batch
            req_ids = list(batch.req_ids[:batch.num_reqs])
            counts = [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids]
            assert sum(counts) == scheduler_output.total_num_scheduled_tokens
            assert len(self._smoke_pending_residuals) == self._smoke_residual_family_count
            for name, layer, tensor in self._smoke_pending_residuals:
                assert tensor.shape[0] >= sum(counts)
                offset = 0
                for rid, count in zip(req_ids, counts):
                    start = self._smoke_computed.get(rid, 0)
                    self._smoke_residuals.append({
                        "request_id": rid, "act_name": name, "layer_no": layer,
                        "start": start, "end": start + count,
                        "tensor": tensor[offset:offset + count].clone(),
                    })
                    offset += count
            for rid, count in zip(req_ids, counts):
                self._smoke_computed[rid] = self._smoke_computed.get(rid, 0) + count
            self._smoke_pending_residuals.clear()
        return result

    def smoke_start(self):
        self._smoke_active = True

    def smoke_describe_runner(self):
        return type(self.model_runner).__module__

    def smoke_dump(self):
        self._smoke_active = False
        path = Path(os.environ["DMI_SMOKE_LOGITS_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._smoke_logits, path)
        if self._smoke_residual_enabled:
            torch.save(self._smoke_residuals, os.environ["DMI_SMOKE_RESIDUALS_PATH"])


class StockV1Worker(_LogitTap, Worker):
    pass


class StockV2Worker(_LogitTap, Worker):
    pass


class MonitoredV1Worker(_LogitTap, DMXGPUWorker):
    pass


class MonitoredV2Worker(_LogitTap, DMXV2GPUWorker):
    pass
