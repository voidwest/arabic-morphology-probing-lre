from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable


def _gpu() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=3)
        if result.returncode != 0 or not result.stdout.strip():
            return {"available": False, "reason": result.stderr.strip() or "nvidia-smi failed"}
        fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        names = ["name", "utilization_gpu_percent", "memory_used_mib", "memory_total_mib", "power_watts", "temperature_c"]
        value: dict[str, Any] = {"available": True}
        for name, field in zip(names, fields):
            value[name] = float(field) if name != "name" else field
        return value
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def _host_memory() -> dict[str, Any]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    except Exception:
        return {}
    return {"total_mib": values.get("MemTotal", 0) / 1024, "available_mib": values.get("MemAvailable", 0) / 1024}


class Telemetry:
    def __init__(self, path: Path, *, run_root: Path, counter: Callable[[], int], interval_seconds: float = 15.0):
        self.path = path
        self.run_root = run_root
        self.counter = counter
        self.interval_seconds = interval_seconds
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        usage = shutil.disk_usage(self.run_root)
        elapsed = max(time.monotonic() - self.started, 1e-6)
        record = {
            "timestamp_epoch": time.time(),
            "examples_processed": self.counter(),
            "examples_per_sec": self.counter() / elapsed,
            "elapsed_seconds": elapsed,
            "disk_free_bytes": usage.free,
            "disk_used_bytes": usage.used,
            "host_memory": _host_memory(),
            "gpu": _gpu(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sample()
        self._thread = threading.Thread(target=self._run, name="bf16-telemetry", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._sample()
