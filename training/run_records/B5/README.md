# B5 run record: QAD R
Host node1, systemd unit qad-qadmix-chain; arm R ran 2026-10-22 17:26Z to 22:49Z on GPUs 0-3, MASTER_PORT 29661.
Main command line: `launch/chain.sh` (waits for B4, then train_qad.sh --mode train ... --kl reverse for R; it also ran arm F, which is not used). It equals the released work/qwen38fn_qad/probes/student_traj_1022/train/qadmix_chain.sh except for the home path.
`config/config.json` and `config/plan.json` are R's resolved trainer config and plan; output $ARC_RAM_RUNS/qwen38fn_qad/qadmix-R-store/export/v001000.
