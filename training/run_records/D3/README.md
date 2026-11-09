# D3 run record: ARC-AGI-3 RL v8
Host node2. Units started 2026-10-30 04:25Z to 04:32Z and stopped 13:48Z; topology tp1x4 over GPUs 0-7, head ce7b463ff on branch arc3-v8.
Main command lines: `launch/rl-v8-*.journal.txt` (orchestrator.py --config, replica_control.py, group_pack.py, launch_rollout_server.sh, train_rl_service.sh), from the released work/qwen38fn_rl/probes/arc3v8_20261029/launch-v8.sh. `launch/v8-start.txt`: fork lock f9cf529b, serve profile, rollout sampling, drafter, base lineage.
Resolved configs: `config/config.json`, `config/plan.json`, `config/harness-{train,eval}.json`, `config/trainer-config.json`; task lists `lists/train.txt`, `lists/heldout.txt`.
The trainer ran the released work/qwen38fn_lora code except the files in `code/`, which are the versions as run. Not in this tree: the rollout fork tree sglang-arc-2-topp.
