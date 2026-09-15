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

    def smoke_start(self):
        self._smoke_active = True

    def smoke_describe_runner(self):
        return type(self.model_runner).__module__

    def smoke_dump(self):
        self._smoke_active = False
        path = Path(os.environ["DMI_SMOKE_LOGITS_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._smoke_logits, path)


class StockV1Worker(_LogitTap, Worker):
    pass


class StockV2Worker(_LogitTap, Worker):
    pass


class MonitoredV1Worker(_LogitTap, DMXGPUWorker):
    pass


class MonitoredV2Worker(_LogitTap, DMXV2GPUWorker):
    pass
