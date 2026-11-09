"""SGLang worker startup for the MTP dense10240 experiment: run the INT3 plugin's sitecustomize (this directory sits
before it on PYTHONPATH and would otherwise shadow it), then install mtpd_runtime when ARC_MTPD=1.
Any failure stops the interpreter (Python would otherwise print and ignore a sitecustomize exception)."""
import os

try:
    chain = os.environ.get("ARC_MTPD_CHAIN_SITECUSTOMIZE")
    if os.environ.get("ARC_MTPD") == "1" and not chain:
        raise RuntimeError("ARC_MTPD=1 needs ARC_MTPD_CHAIN_SITECUSTOMIZE (the INT3 plugin's sitecustomize.py)")
    if chain:
        import runpy
        runpy.run_path(chain, run_name="int3_sitecustomize")
    if os.environ.get("ARC_MTPD") == "1":
        from mtpd_runtime import install
        install()
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(71)
