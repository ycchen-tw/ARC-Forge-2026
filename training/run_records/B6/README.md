# B6 run record: materialize QAD R and upload Kaggle v3
Host node1, 2026-10-23.
`launch/materialize.sh R` (materialize_package.py) wrote $ARC_RAM_RUNS/qwen38fn_qad/models/qadmix-R-v1000 from the A5 package and the B5 export; it equals the released work/qwen38fn_qad/probes/student_traj_1022/train/materialize.sh except for how the roots are written: `$ARC_RAM_RUNS` / `$ARC_RUNS` here, `/dev/shm/uid-$UID/arcforge-runs` / `$HOME/work/runs` in the released script (the same directories on node1).
`launch/upload_par.sh` runs `launch/upload_version_parallel.py`, which uploaded $ARC_RAM_RUNS/qwen38fn_qad/kaggle-upload-qadmix-R as threerabbits/qwen38fn-gean320-int3-gptq/Transformers/default/3 (2026-10-23 17:47Z to 17:54Z).
Not recorded: the build_qad_bf16dense.py invocation that made the BF16-dense twin.
