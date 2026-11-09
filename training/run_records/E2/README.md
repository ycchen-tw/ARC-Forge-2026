# E2 run record: DFlash data plans and target profiles
Host node1, 2026-10-12 to 2026-10-19, run dirs $ARC_RUNS/qwen38fn_dflash/<date>-<name>.
`config/<plan>.plan_meta.json` holds each make_data_plan.py argv (including the --bench ledgers) and `config/<plan>.summary.json` its totals: int3-plan (10-12), codex1015-plan, codex1015-small-plan (10-17), codex1015-small50-plan (10-18), l8-plan (10-19).
`config/target-profile-1012.json` / `target-profile-1017.json` are the INT3 W3A8 TP1 target profiles the captures used (server launch command included).
