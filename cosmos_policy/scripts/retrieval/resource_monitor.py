"""Sample total GPU and registered retrieval-process memory with NVML."""
import json
from pathlib import Path
import threading
import time


class ResourceMonitor(threading.Thread):
    def __init__(self, gpu, output, interval=0.2):
        super().__init__(daemon=True)
        self.gpu = gpu
        self.output = Path(output)
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples = []
        self.hardware = {}
        self.errors = []
        self.registered = {}

    def run(self):
        try:
            import pynvml as nv
            nv.nvmlInit()
            handle = (nv.nvmlDeviceGetHandleByIndex(int(self.gpu)) if str(self.gpu).isdigit()
                      else nv.nvmlDeviceGetHandleByUUID(self.gpu))
            decode = lambda value: value.decode() if isinstance(value, bytes) else value
            total = nv.nvmlDeviceGetMemoryInfo(handle).total
            self.hardware = dict(name=decode(nv.nvmlDeviceGetName(handle)),
                                 uuid=decode(nv.nvmlDeviceGetUUID(handle)), total_bytes=total,
                                 driver=decode(nv.nvmlSystemGetDriverVersion()), interval_seconds=self.interval)
            registry = self.output / 'retrieval_resources.json'
            registry_mtime = None
            start = time.perf_counter()
            with (self.output / 'gpu_samples.jsonl').open('x') as f:
                while not self.stop_event.is_set():
                    try:
                        if registry.exists() and registry.stat().st_mtime_ns != registry_mtime:
                            self.registered = json.loads(registry.read_text())
                            registry_mtime = registry.stat().st_mtime_ns
                        memory = nv.nvmlDeviceGetMemoryInfo(handle)
                        processes = {p.pid: int(p.usedGpuMemory) for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)
                                     if 0 <= p.usedGpuMemory <= total}
                        workers = self.registered.get('workers', {})
                        roles = {role: processes.get(worker['pid'], 0) for role, worker in workers.items()}
                        sample = dict(seconds=time.perf_counter() - start, gpu_used_bytes=memory.used,
                                      process_bytes=processes, retrieval_bytes=sum(roles.values()),
                                      worker_bytes=roles, utilization=nv.nvmlDeviceGetUtilizationRates(handle).gpu,
                                      sm_clock_mhz=nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM))
                        self.samples.append(sample)
                        f.write(json.dumps(sample) + '\n')
                        f.flush()
                    except Exception as exc:
                        self.errors.append(str(exc))
                    self.stop_event.wait(self.interval)
            nv.nvmlShutdown()
        except Exception as exc:
            self.errors.append(str(exc))

    def finish(self):
        self.stop_event.set()
        self.join(timeout=6)
        return dict(hardware=self.hardware, sample_count=len(self.samples), errors=self.errors[:10],
                    gpu_peak_used_mib=max((s['gpu_used_bytes'] for s in self.samples), default=0) / 2**20,
                    retrieval_process_peak_mib=max((s['retrieval_bytes'] for s in self.samples), default=0) / 2**20,
                    worker_peak_mib={role: max((s['worker_bytes'].get(role, 0) for s in self.samples), default=0) / 2**20
                                     for role in self.registered.get('workers', {})})
