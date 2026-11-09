"""Install the reviewed INT3 plugin in SGLang's spawned worker interpreters."""
import os

if os.environ.get("ARC_INT3_DEPLOY") == "1":
    try:
        from int3_deploy_runtime import install
        install()
    except BaseException:
        # Python otherwise prints and ignores a sitecustomize exception, leaving
        # workers on the ordinary NVFP4 loader. A bad plugin must stop startup.
        import traceback
        traceback.print_exc()
        os._exit(70)
