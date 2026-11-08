"""Measured cost of one run: peak GPU memory, peak CPU memory, GPU energy (dashboard: Cost & efficiency).

    meter = RunMeter(dev)                 # at the run's t0
    ...
    metrics["cost"] = meter.result()      # next to train_sec

Energy comes from the GPU's own counter (NVML total energy, Volta and newer). On a MIG slice the counter belongs to
the whole physical GPU, which other slices share, so it is scaled by the slice fraction (x / 7 for xg.yygb) and
marked "nvml_slice_share". measured_sec is the span the counter covers: a resumed run only measures its last
segment, and the dashboard scales the energy up to train_sec. Peak CPU memory is the process's peak resident set,
so in a process that trains several runs one after another it is an upper bound for the later runs.
Every part is optional and never fatal: a missing value is left out and the dashboard estimates it from time.
"""
import os
import resource
import time

from sci_jepa.infra.report import _mig_fraction


def _nvml(dev):
    """NVML handle of the physical GPU behind torch device `dev`, its name and the slice fraction (1 = full GPU)."""
    import pynvml as N
    N.nvmlInit()
    ids = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    i = dev.index or 0
    key = ids[i] if i < len(ids) else str(i)
    h = N.nvmlDeviceGetHandleByUUID(key) if key.startswith(("GPU-", "MIG-")) else N.nvmlDeviceGetHandleByIndex(int(key))
    name = N.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name
    if key.startswith("MIG-"):
        return N, N.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(h), name, _mig_fraction(name)
    return N, h, name, 1.0


class RunMeter:
    def __init__(self, dev):
        self.t0, self.dev, self.N = time.time(), dev, None
        self.device, self.slice_fraction, self.e0 = None, None, None
        if getattr(dev, "type", None) != "cuda":
            return
        try:
            import torch
            torch.cuda.reset_peak_memory_stats(dev)
            self.device = torch.cuda.get_device_name(dev)
        except Exception:                               # noqa: BLE001
            pass
        try:
            self.N, self.h, name, self.slice_fraction = _nvml(dev)
            self.device = name or self.device
            self.e0 = self.N.nvmlDeviceGetTotalEnergyConsumption(self.h)    # mJ
        except Exception:                               # noqa: BLE001  (no NVML or no energy counter)
            self.e0 = None

    def result(self):
        out = {"measured_sec": round(time.time() - self.t0, 1), "device": self.device,
               "slice_fraction": self.slice_fraction,
               "cpu_mem_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9, 2)}
        if getattr(self.dev, "type", None) == "cuda":
            try:
                import torch
                out["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated(self.dev) / 1e9, 2)
            except Exception:                           # noqa: BLE001
                pass
        if self.e0 is not None:
            try:
                kwh = (self.N.nvmlDeviceGetTotalEnergyConsumption(self.h) - self.e0) / 3.6e9
                out["gpu_kwh"] = kwh * self.slice_fraction
                out["energy_source"] = "nvml" if self.slice_fraction == 1.0 else "nvml_slice_share"
            except Exception:                           # noqa: BLE001
                pass
        return {k: v for k, v in out.items() if v is not None}
