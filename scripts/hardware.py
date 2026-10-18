import ctypes
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("load", ctypes.c_ulong),
        ("total", ctypes.c_ulonglong),
        ("available", ctypes.c_ulonglong),
        ("page", ctypes.c_ulonglong),
        ("apage", ctypes.c_ulonglong),
        ("virtual", ctypes.c_ulonglong),
        ("avirtual", ctypes.c_ulonglong),
        ("extended", ctypes.c_ulonglong),
    ]


def main():
    m = MemoryStatus()
    m.length = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    ) as key:
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    result = {
        "cpu": cpu,
        "threads": os.cpu_count(),
        "ram_gb": round(m.total / 2**30, 1),
        "available_ram_gb": round(m.available / 2**30, 1),
        "free_disk_gb": round(shutil.disk_usage(".").free / 2**30, 1),
        "python": platform.python_version(),
    }
    try:
        result["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        result["gpu"] = None
    root = Path(__file__).resolve().parents[1]
    (root / "models").mkdir(exist_ok=True)
    (root / "models/hardware.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    temp = Path(tempfile.gettempdir())
    print("temp", str(temp))
    for folder in temp.glob("pip-*"):
        for p in folder.rglob("*"):
            if p.is_file() and p.stat().st_size > 5_000_000:
                print("download", p.name, round(p.stat().st_size / 2**20, 1), "MB")


if __name__ == "__main__":
    main()
