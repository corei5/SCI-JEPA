"""Live infrastructure monitor: one sample per second of the job's GPUs, CPU, memory and training progress.

One Monitor runs per job (in the dashboard server process: pipeline --stage serve, or the single-process run).
It writes
    <results>/dashboard/infra/live.json   latest sample + the last `keep` samples (the dashboard's infra.html)
    <results>/infra/<job>/timeseries.csv  every sample, for the report's timelines
    <results>/infra/<job>/system.json     sysinfo record of the job
Training processes report their progress through write_progress() (called by train_one about every 5 s).

On MIG slices NVML reports memory per slice, but no utilization per slice; power and temperature belong to the
whole physical GPU, which other jobs' slices share. Both are labelled that way, never attributed to one run.
"""
import collections
import csv
import glob
import json
import os
import re
import threading
import time

from sci_jepa.infra import sysinfo


def _atomic(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, path)


def progress_dir(results_dir):
    return os.path.join(results_dir, "infra", "progress")


def write_progress(results_dir, run, **fields):
    """A training process's latest progress (tiny JSON, atomic). run: the run's name (dashboard name)."""
    safe = re.sub(r"[^A-Za-z0-9_.@=+-]", "_", run)
    _atomic({"run": run, "pid": os.getpid(), "time": time.time(), "job": os.environ.get("SLURM_JOB_ID"),
             "shard": os.environ.get("SCI_JEPA_SHARD"), **fields},
            os.path.join(progress_dir(results_dir), f"{safe}.json"))


def _slurm_end():
    """Unix time the job ends (scontrol EndTime), or None."""
    end = sysinfo._slurm().get("EndTime")
    try:
        return time.mktime(time.strptime(end, "%Y-%m-%dT%H:%M:%S")) if end else None
    except ValueError:
        return None


class Monitor(threading.Thread):
    def __init__(self, results_dir, interval=1.0, keep=600):
        super().__init__(daemon=True)
        self.results, self.interval, self.keep = results_dir, interval, keep
        self.job = os.environ.get("SLURM_JOB_ID") or f"local-{os.uname().nodename}-{os.getpid()}"
        self.dir = os.path.join(results_dir, "infra", self.job)
        # the dashboard server serves <results>/dashboard/: the live file goes there (infra.html polls it)
        self.live = os.path.join(results_dir, "dashboard", "infra", "live.json")
        os.makedirs(self.dir, exist_ok=True)
        self.system = sysinfo.collect(results_dir)
        sysinfo.write(self.system, os.path.join(self.dir, "system.json"))
        self.t_start, self.t_end = time.time(), _slurm_end()
        self.hist = collections.deque(maxlen=keep)
        self._stop = threading.Event()
        self._nvml()
        self._procs, self._procs_t = [], 0.0
        self._runs_done, self._runs_t = None, 0.0
        self.csv_path = os.path.join(self.dir, "timeseries.csv")
        self._csv_cols = None

    # ---- GPUs -------------------------------------------------------------------------------------------
    def _nvml(self):
        self.N, self.phys, self.migs = None, [], []
        try:
            import pynvml as N
            N.nvmlInit()
            self.N = N
            for i in range(N.nvmlDeviceGetCount()):
                h = N.nvmlDeviceGetHandleByIndex(i)
                try:
                    mig_on = N.nvmlDeviceGetMigMode(h)[0] == 1
                except N.NVMLError:
                    mig_on = False
                self.phys.append({"i": i, "h": h, "name": N.nvmlDeviceGetName(h), "mig": mig_on})
                if mig_on:
                    for j in range(N.nvmlDeviceGetMaxMigDeviceCount(h)):
                        try:
                            m = N.nvmlDeviceGetMigDeviceHandleByIndex(h, j)
                        except N.NVMLError:
                            continue
                        self.migs.append({"parent": i, "h": m, "uuid": N.nvmlDeviceGetUUID(m),
                                          "name": N.nvmlDeviceGetName(m)})
        except Exception as e:                          # noqa: BLE001  (no NVML: CPU and progress only)
            self.nvml_error = str(e)

    def _gpu_sample(self):
        N, out = self.N, []
        if N is None:
            return out
        def tryv(f, *a):
            try:
                return f(*a)
            except N.NVMLError:
                return None
        for g in self.phys:
            h = g["h"]
            util = None if g["mig"] else tryv(N.nvmlDeviceGetUtilizationRates, h)
            mem = None if g["mig"] else tryv(N.nvmlDeviceGetMemoryInfo, h)
            p = tryv(N.nvmlDeviceGetPowerUsage, h)
            out.append({"kind": "gpu", "id": g["i"], "name": g["name"], "mig_mode": g["mig"],
                        "util_pct": util.gpu if util else None,
                        "mem_used_gb": round(mem.used / 1e9, 2) if mem else None,
                        "mem_total_gb": round(mem.total / 1e9, 1) if mem else None,
                        "power_w": round(p / 1000, 1) if p is not None else None,
                        "temp_c": tryv(N.nvmlDeviceGetTemperature, h, N.NVML_TEMPERATURE_GPU),
                        "sm_clock_mhz": tryv(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_SM)})
        for m in self.migs:
            mem = tryv(N.nvmlDeviceGetMemoryInfo, m["h"])
            out.append({"kind": "mig", "id": m["uuid"], "parent": m["parent"], "name": m["name"],
                        "mem_used_gb": round(mem.used / 1e9, 2) if mem else None,
                        "mem_total_gb": round(mem.total / 1e9, 1) if mem else None})
        return out

    # ---- CPU / memory of the job's processes ------------------------------------------------------------
    def _cpu_sample(self):
        try:
            import psutil
        except ImportError:
            return {}
        now = time.time()
        if now - self._procs_t > 10 or not self._procs:   # the job's process tree, refreshed every 10 s
            try:
                root = psutil.Process(os.getppid())
                procs = [root] + root.children(recursive=True)
            except psutil.Error:
                procs = [psutil.Process()]
            known = {p.pid: p for p in self._procs}
            self._procs = [known.get(p.pid, p) for p in procs]
            self._procs_t = now
        cpu, rss = 0.0, 0
        for p in self._procs:
            try:
                cpu += p.cpu_percent(None)
                rss += p.memory_info().rss
            except psutil.Error:
                pass
        vm = psutil.virtual_memory()
        cores = self.system["cpu"]["cores_allocated"] or 1
        return {"cpu_pct_of_allocated": round(cpu / cores, 1), "cpu_cores_busy": round(cpu / 100, 2),
                "rss_gb": round(rss / 1e9, 2), "node_mem_used_pct": vm.percent, "load1": os.getloadavg()[0]}

    # ---- training progress ------------------------------------------------------------------------------
    def _progress(self):
        rows = []
        for f in glob.glob(os.path.join(progress_dir(self.results), "*.json")):
            try:
                r = json.load(open(f))
            except (OSError, json.JSONDecodeError):
                continue
            if r.get("job") == os.environ.get("SLURM_JOB_ID") and time.time() - r.get("time", 0) < 600:
                rows.append(r)                         # this job's runs, reported in the last 10 minutes
        return sorted(rows, key=lambda r: str(r.get("shard")))

    def _runs(self):
        now = time.time()
        if now - self._runs_t > 30:
            plan_p = os.path.join(self.results, "infra", "plan.json")
            plan = json.load(open(plan_p)) if os.path.exists(plan_p) else {}
            done = total = 0
            for exp, keys in (plan.get("experiments") or {}).items():
                total += len(keys)
                done += sum(os.path.exists(os.path.join(self.results, exp, f"{k}.json")) for k in keys)
            self._runs_done = {"done": done, "total": total} if total else None
            self._runs_t = now
        return self._runs_done

    # ---- loop -------------------------------------------------------------------------------------------
    def sample(self):
        now = time.time()
        return {"t": now, "time": time.strftime("%H:%M:%S", time.localtime(now)),
                "elapsed_s": round(now - self.t_start), "left_s": round(self.t_end - now) if self.t_end else None,
                "gpus": self._gpu_sample(), "cpu": self._cpu_sample(), "progress": self._progress(), "runs": self._runs()}

    def _csv(self, s):
        row = {"t": round(s["t"], 1), "elapsed_s": s["elapsed_s"]}
        for g in s["gpus"]:
            key = f"{g['kind']}{g['id'] if g['kind'] == 'gpu' else g['id'][-8:]}"
            for k in ("mem_used_gb", "power_w", "temp_c", "util_pct"):
                if g.get(k) is not None:
                    row[f"{key}_{k}"] = g[k]
        row.update({f"cpu_{k}": v for k, v in s["cpu"].items()})
        row["steps_per_s_total"] = round(sum(p.get("steps_per_s") or 0 for p in s["progress"]), 3)
        row["runs_training"] = sum(p.get("status") == "training" for p in s["progress"])
        new = not os.path.exists(self.csv_path)
        if self._csv_cols is None or set(row) - set(self._csv_cols):
            self._csv_cols = list(dict.fromkeys((self._csv_cols or []) + list(row)))
            if not new:                                # a new column appeared: start a continuation file
                os.replace(self.csv_path, self.csv_path.replace(".csv", f".{int(s['t'])}.csv"))
                new = True
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_cols)
            if new:
                w.writeheader()
            w.writerow(row)

    def brief(self):
        s = self.system
        return {"job": self.job, "host": s["host"], "partition": s["slurm"].get("SLURM_JOB_PARTITION"),
                "time_limit": s["slurm"].get("TimeLimit"), "gpus": s["gpus"].get("nvidia_smi_L"),
                "driver": s["gpus"].get("driver"), "cuda": s["torch"].get("cuda_build"),
                "torch": s["torch"].get("torch"), "cpu": s["cpu"], "started": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(self.t_start))}

    def run(self):
        brief = self.brief()
        while not self._stop.is_set():
            t = time.time()
            try:
                s = self.sample()
                self.hist.append(s)
                self._csv(s)
                _atomic({"system": brief, "now": s, "history": list(self.hist), "interval_s": self.interval},
                        self.live)
            except Exception as e:                      # noqa: BLE001  (monitoring never stops a run)
                print(f"[infra] sample failed: {e}", flush=True)
            self._stop.wait(max(0.05, self.interval - (time.time() - t)))

    def stop(self):
        self._stop.set()


def start(results_dir, interval=1.0):
    """Start the monitor thread for this job; returns it (call .stop() at the end, or let the process exit)."""
    m = Monitor(results_dir, interval)
    m.start()
    print(f"[infra] monitoring every {interval:g} s -> {m.dir} (live: {m.live})", flush=True)
    return m
