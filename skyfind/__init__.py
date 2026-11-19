"""skyfind: where and when a star-field timelapse was shot, and which
satellites crossed it."""

import multiprocessing as _mp
import sys as _sys

# Worker processes start fresh everywhere, as Windows starts them: a fork
# copies a parent that may already hold the GPU (CUDA can't be used in a
# forked child) and a solver pool's threads. Linux and macOS (the
# container) follow; the engine's pools, made after this import, too.
if _sys.platform != "win32" and _mp.get_start_method(allow_none=True) is None:
    _mp.set_start_method("spawn")
