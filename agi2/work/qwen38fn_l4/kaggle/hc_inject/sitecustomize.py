"""Kaggle v15: hook arc_hostcomm_hook in after import, without modifying the pinned sglang source.

Vast's approach appends an import to the end of dp_attention.py in the r9-r2 source (make_hc_runtime.sh), but
Kaggle's candidate source is sha-verified file by file, so modifying it means redoing the whole candidate packaging.
This uses sitecustomize instead: the builder puts this directory first on the server's PYTHONPATH, so every sglang
subprocess (a freshly spawned interpreter) runs this file at start-up; it puts a finder at the front of
sys.meta_path that calls `arc_hostcomm_hook.install(module)` once `sglang.srt.layers.dp_attention` has executed --
the same point as an import appended to the end of that file.

Active only with ARC_HOSTCOMM=1. If another sitecustomize is on the path, it still runs afterwards rather than being
swallowed.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

TARGET = "sglang.srt.layers.dp_attention"


class _HostcommFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)  # intercept only once; later find_spec calls take the normal path
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            raise ImportError(f"ARC_HOSTCOMM: {name} not found on sys.path")
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            import arc_hostcomm_hook
            arc_hostcomm_hook.install(module)

        spec.loader.exec_module = exec_module
        return spec


def _chain_other_sitecustomize():
    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _main():
    if os.environ.get("ARC_HOSTCOMM") == "1":
        if not os.environ.get("ARC_HC_LIB"):
            raise RuntimeError("ARC_HOSTCOMM=1 requires ARC_HC_LIB")
        sys.meta_path.insert(0, _HostcommFinder())
    _chain_other_sitecustomize()


try:
    _main()
except BaseException:
    # site.py only prints "Error in sitecustomize" and carries on, so the hook silently goes missing; end the
    # process here instead.
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(97)
