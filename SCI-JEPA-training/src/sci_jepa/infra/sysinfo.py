"""One record of the machine a job runs on: hardware, drivers, Slurm allocation and software versions.

    from sci_jepa.infra.sysinfo import collect, write
    write(collect(), "runs/paper_a_results/infra/<job>/system.json")
"""
import json
import os
import platform
import socket
import subprocess
import sys
import time


def _run(cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu():
    model = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_gb = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem_gb = round(int(line.split()[1]) / 1024 ** 2, 1)
                break
    except OSError:
        pass
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:
        allowed = os.cpu_count()
    return {"model": model, "cores_node": os.cpu_count(), "cores_allocated": allowed, "ram_node_gb": mem_gb}


def _gpus():
    """Physical GPUs and MIG slices visible to the job (NVML; falls back to nvidia-smi)."""
    out = {"driver": None, "cuda_driver": None, "physical": [], "mig": []}
    try:
        import pynvml as N
        N.nvmlInit()
        out["driver"] = N.nvmlSystemGetDriverVersion()
        v = N.nvmlSystemGetCudaDriverVersion_v2()
        out["cuda_driver"] = f"{v // 1000}.{(v % 1000) // 10}"
        for i in range(N.nvmlDeviceGetCount()):
            h = N.nvmlDeviceGetHandleByIndex(i)
            try:
                mig_on = N.nvmlDeviceGetMigMode(h)[0] == 1
            except N.NVMLError:
                mig_on = False
            mem = N.nvmlDeviceGetMemoryInfo(h)
            try:
                plim = N.nvmlDeviceGetPowerManagementLimit(h) / 1000
            except N.NVMLError:
                plim = None
            out["physical"].append({"index": i, "name": N.nvmlDeviceGetName(h), "uuid": N.nvmlDeviceGetUUID(h),
                                    "memory_gb": round(mem.total / 1e9, 1), "mig_mode": mig_on, "power_limit_w": plim})
            if mig_on:
                for j in range(N.nvmlDeviceGetMaxMigDeviceCount(h)):
                    try:
                        m = N.nvmlDeviceGetMigDeviceHandleByIndex(h, j)
                    except N.NVMLError:
                        continue
                    mm = N.nvmlDeviceGetMemoryInfo(m)
                    out["mig"].append({"parent": i, "uuid": N.nvmlDeviceGetUUID(m), "name": N.nvmlDeviceGetName(m),
                                       "memory_gb": round(mm.total / 1e9, 1)})
        N.nvmlShutdown()
    except Exception as e:                              # noqa: BLE001  (no NVML: nvidia-smi or nothing)
        out["nvml_error"] = str(e)
        txt = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        rows = [[x.strip() for x in l.split(",")] for l in txt.splitlines() if l.count(",") == 2]   # skip error text
        for i, (name, drv, mem) in enumerate(rows):
            out["driver"] = drv
            out["physical"].append({"index": i, "name": name, "memory": mem})
    out["nvidia_smi_L"] = _run(["nvidia-smi", "-L"])
    return out


def _torch():
    try:
        import torch
        devs = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devs.append({"index": i, "name": p.name, "memory_gb": round(p.total_memory / 1e9, 1),
                         "compute_capability": f"{p.major}.{p.minor}", "multiprocessors": p.multi_processor_count})
        return {"torch": torch.__version__, "cuda_build": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                "cuda_devices_visible": devs, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    except Exception as e:                              # noqa: BLE001
        return {"error": str(e)}


def _versions():
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for pkg in ("sci-jepa", "torch", "transformers", "sentence-transformers", "safetensors", "numpy", "scipy",
                "scikit-learn", "pandas", "pyarrow", "accelerate", "nvidia-ml-py", "matplotlib"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _slurm():
    env = {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT",
                                       "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
                                       "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_QOS") if k in os.environ}
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        txt = _run(["scontrol", "show", "job", job])
        for key in ("TimeLimit", "StartTime", "EndTime", "TRES", "Partition", "QOS"):
            for part in txt.split():
                if part.startswith(key + "="):
                    env[key] = part.split("=", 1)[1]
                    break
    return env


def _storage(path):
    path = path or os.getcwd()
    txt = _run(["df", "-hT", path]).splitlines()
    return {"path": path, "df": txt[-1] if txt else ""}


def collect(results_dir=None):
    """Everything about the machine and software of this job, as one JSON-ready dict."""
    return {"collected": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
            "os": f"{platform.system()} {platform.release()}", "cpu": _cpu(), "gpus": _gpus(), "torch": _torch(),
            "versions": _versions(), "slurm": _slurm(),
            "storage": _storage(results_dir or os.environ.get("PAPER_A_OUT"))}


def write(rec, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    json.dump(rec, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, path)
    return path


if __name__ == "__main__":
    print(json.dumps(collect(), indent=1, default=str))
