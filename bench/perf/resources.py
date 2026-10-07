"""Ressources locales pendant une mesure : VRAM, GPU, RAM, et processus du serveur.

Une mesure de débit sans la mémoire correspondante n'est pas exploitable : c'est le couple
(débit, VRAM crête) qui décide si une configuration tient sur la machine. Ce module fournit les deux
sources disponibles sans dépendance obligatoire :

- **GPU / VRAM** : `nvidia-smi` (`--query-gpu`, format CSV) — présent sur toute machine NVIDIA ;
- **RAM** : `psutil` s'il est installé, sinon `GlobalMemoryStatusEx` (Windows) ou `/proc/meminfo`
  (Linux), en `ctypes`/lecture de fichier.

Toute sonde indisponible retourne `None` et la raison est conservée : une campagne ne doit jamais
échouer parce qu'un compteur manque.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time

GPU_FIELDS = ("memory.used", "memory.total", "utilization.gpu", "temperature.gpu", "power.draw")
SAMPLE_INTERVAL_S = 0.5


def nvidia_smi_path() -> str | None:
    return shutil.which("nvidia-smi")


def sample_gpu(index: int = 0, smi: str | None = None) -> dict | None:
    """Un relevé GPU. `None` si `nvidia-smi` est absent ou échoue."""
    smi = smi or nvidia_smi_path()
    if not smi:
        return None
    cmd = [smi, f"--id={index}", f"--query-gpu={','.join(GPU_FIELDS)}",
           "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    values = [v.strip() for v in proc.stdout.strip().splitlines()[0].split(",")]
    if len(values) != len(GPU_FIELDS):
        return None
    out: dict = {}
    for name, value in zip(GPU_FIELDS, values):
        key = {"memory.used": "vram_used_mib", "memory.total": "vram_total_mib",
               "utilization.gpu": "gpu_util_pct", "temperature.gpu": "gpu_temp_c",
               "power.draw": "gpu_power_w"}[name]
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = None
    return out


def gpu_name(index: int = 0, smi: str | None = None) -> str | None:
    smi = smi or nvidia_smi_path()
    if not smi:
        return None
    try:
        proc = subprocess.run([smi, f"--id={index}", "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or None


def system_ram() -> dict | None:
    """{total_mib, available_mib} sans dépendance obligatoire."""
    try:  # psutil si présent
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return {"total_mib": int(vm.total / 1048576), "available_mib": int(vm.available / 1048576),
                "source": "psutil"}
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return None
            return {"total_mib": int(stat.ullTotalPhys / 1048576),
                    "available_mib": int(stat.ullAvailPhys / 1048576), "source": "GlobalMemoryStatusEx"}
        except (OSError, AttributeError):
            return None
    try:
        info = {}
        for line in open("/proc/meminfo", encoding="utf-8"):
            key, _, rest = line.partition(":")
            info[key.strip()] = int(rest.strip().split()[0])
        return {"total_mib": info["MemTotal"] // 1024, "available_mib": info["MemAvailable"] // 1024,
                "source": "/proc/meminfo"}
    except (OSError, KeyError, ValueError, IndexError):
        return None


def process_rss_mib() -> int | None:
    """RSS du processus courant (le serveur lancé est son enfant sur la plupart des moteurs)."""
    try:
        import psutil  # type: ignore
        return int(psutil.Process().memory_info().rss / 1048576)
    except ImportError:
        return None


class ResourceSampler:
    """Échantillonne VRAM/GPU/RAM en tâche de fond et conserve les **pics** de la fenêtre mesurée.

    Les pics sont ce qui compte : une VRAM moyenne sur une génération ne dit rien, la crête dit si la
    configuration tient. Utilisable comme gestionnaire de contexte ou avec `start()`/`stop()`.
    """

    def __init__(self, gpu_index: int = 0, interval: float = SAMPLE_INTERVAL_S) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = sample_gpu(self.gpu_index) or {}
            ram = system_ram() or {}
            self.samples.append({**sample, "ram_available_mib": ram.get("available_mib"),
                                 "t": time.perf_counter()})
            self._stop.wait(self.interval)

    def start(self) -> "ResourceSampler":
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 2))
            self._thread = None
        return self.peak()

    def peak(self) -> dict:
        """Pics observés. Chaque clé est `None` si aucun échantillon ne la portait."""
        def top(key: str):
            vals = [s[key] for s in self.samples if isinstance(s.get(key), (int, float))]
            return max(vals) if vals else None

        def bottom(key: str):
            vals = [s[key] for s in self.samples if isinstance(s.get(key), (int, float))]
            return min(vals) if vals else None

        return {"samples": len(self.samples),
                "vram_peak_mib": top("vram_used_mib"),
                "vram_end_mib": self.samples[-1].get("vram_used_mib") if self.samples else None,
                "vram_total_mib": top("vram_total_mib"),
                "gpu_util_peak_pct": top("gpu_util_pct"),
                "gpu_temp_peak_c": top("gpu_temp_c"),
                "gpu_power_peak_w": top("gpu_power_w"),
                "ram_min_available_mib": bottom("ram_available_mib")}

    def __enter__(self) -> "ResourceSampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def host_summary(gpu_index: int = 0) -> dict:
    """Ce que la machine est, pour que les chiffres publiés soient interprétables."""
    return {"gpu": gpu_name(gpu_index), "ram": system_ram(),
            "nvidia_smi": nvidia_smi_path(), "python": sys.version.split()[0],
            "platform": sys.platform}
