# A4 run record: one-shot INT3 GPTQ of the gEAN320 experts
Run 20261009-gean320-int3-mm-2a34605e on node2, 2026-10-09 15:27Z to 16:27Z, GPUs 0-3, calibration server port 31073.
Main command line: `launch/launch.sh` (run_gean_int3_mm.py run); `launch/events.jsonl` records every child command the orchestrator ran. `config/source_commit.txt` and `config/identity.json` identify the code and run.
It reuses inputs of the stopped first attempt 01a1206d: `config/pruning_calibration.json`, `config/recorded-config-99c11efb.json`, its FP8 calibration model, masks and the sglang checkout at fork cd15559a (not in this tree).
`lists/calib-selection.json` lists the 50 calibration segments (2,004,330 tokens); `config/gptq_calibration.json` and `config/w*-complete.json` identify the per-window outputs under parts/.
Not recorded: the command that built 01a1206d's calib-fp8 model (the FP8 model pruned to gEAN320 that served calibration).
