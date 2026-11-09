# A5 run record: INT3 deploy package and upload
Host node2, 2026-10-09, run dir $ARC_RUNS/qwen38fn_eval/20261009-gean320-int3-deploy.
`launch/post_gptq.sh` exported the sidecar at 16:27Z (export_int3_sidecar.py; its own build step stopped on a path check). `launch/build.sh` (build_int3_deploy.py) then built model-final at 16:49Z to 16:52Z, and `launch/upload_hf.py <model-final> ycchen/qwen38fn-gean320-int3-gptq` uploaded it at 16:56Z.
`launch/rebuild.sh` rebuilds the package from parts/ and checks it against HF's deploy_manifest.json. The Kaggle copy (threerabbits/qwen38fn-gean320-int3-gptq version 1) is model-kaggle: `launch/rebuild_kaggle.sh` exported again at 17:52Z and its build stopped on a full disk; the second build (17:56Z to 17:59Z) has no recorded command.
`config/source_int3_manifest.json` is the sidecar manifest shipped in the package. validation_notes.md (passed to build.sh --validation-notes) is not included.
