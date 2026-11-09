"""Launcher: int3_deploy_runtime.py launch, with this patch directory put first on the workers' PYTHONPATH.

    python mtpd_launch.py launch --model-path <INT3 package> [server arguments]

int3_deploy_runtime.main() prepends its own sitecustomize directory just before runpy.run_module(sglang.launch_server);
the wrapper around runpy.run_module prepends ours after that, so spawned workers run ours, which chains the INT3 one.
"""
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ["ARC_ROOT"])
sys.path.insert(0, str(REPO / "work" / "qwen38fn_eval"))
import int3_deploy_runtime  # noqa: E402

os.environ["ARC_MTPD_CHAIN_SITECUSTOMIZE"] = str(REPO / "work/qwen38fn_eval/int3_deploy_patch/sitecustomize.py")
_run_module = runpy.run_module


def run_module(*args, **kwargs):
    if os.environ.get("ARC_INT3_DEPLOY") != "1":
        raise RuntimeError("int3_deploy_runtime did not set ARC_INT3_DEPLOY before launching")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE), os.environ["PYTHONPATH"]])
    sys.path.insert(0, str(HERE))
    if os.environ.get("ARC_MTPD") == "1":
        from mtpd_runtime import install
        install()
    print(f"ARC_MTPD_LAUNCH PYTHONPATH={os.environ['PYTHONPATH']}", flush=True)
    return _run_module(*args, **kwargs)


# Guarded: SGLang's spawned workers re-import this file as __mp_main__ and must not launch again.
if __name__ == "__main__":
    runpy.run_module = run_module
    int3_deploy_runtime.main()
