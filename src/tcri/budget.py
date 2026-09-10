"""Keep a training run from taking over the machine it runs on.

Four levers: process priority, thread count, a VRAM ceiling, and a duty cycle
that sleeps between batches. The duty cycle matters most on a laptop -- it keeps
the fans down, and a thermally throttled GPU ends up no faster than a
duty-cycled one.
"""
from __future__ import annotations

import os
import sys
import time

PRIORITY = {"idle": 0x00000040, "below": 0x00004000, "normal": 0x00000020}


def set_process_priority(level: str = "below") -> str:
    """Lower this process's scheduling priority. Returns what was applied."""
    if level == "normal":
        return "normal"
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            # GetCurrentProcess returns a HANDLE-sized pseudo-handle. Without
            # these declarations ctypes assumes a 32-bit int and truncates it,
            # and SetPriorityClass then fails with ERROR_INVALID_HANDLE.
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            k32.SetPriorityClass.restype = ctypes.c_int
            ok = k32.SetPriorityClass(k32.GetCurrentProcess(),
                                      PRIORITY.get(level, PRIORITY["below"]))
            if ok:
                return level
            return f"normal (SetPriorityClass err {ctypes.get_last_error()})"
        os.nice(10 if level == "below" else 19)
        return level
    except Exception as e:                                    # never fatal
        return f"normal ({type(e).__name__})"


def limit_threads(n: int) -> int:
    """Cap CPU threads for torch and the BLAS libraries underneath it."""
    n = max(1, int(n))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
    return n


def limit_vram(fraction: float, device: int = 0) -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "n/a (no cuda)"
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(float(fraction), device)
        return f"{fraction:.0%} of {total/2**30:.1f} GiB"
    except Exception as e:
        return f"unset ({type(e).__name__})"


class DutyCycle:
    """Sleep after each unit of work so sustained load stays near `duty`."""

    def __init__(self, duty: float = 1.0, min_sleep: float = 0.001):
        self.duty = min(max(float(duty), 0.05), 1.0)
        self.min_sleep = min_sleep
        self.slept = 0.0
        self._t = None

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.duty >= 1.0 or self._t is None:
            return False
        work = time.perf_counter() - self._t
        pause = work * (1.0 / self.duty - 1.0)
        if pause > self.min_sleep:
            time.sleep(pause)
            self.slept += pause
        return False


def describe(priority: str, threads: int, vram: str, duty: float,
             workers: int) -> str:
    return (f"resource budget: priority={priority}  threads={threads}  "
            f"vram={vram}  duty={duty:.0%}  loader workers={workers}")
