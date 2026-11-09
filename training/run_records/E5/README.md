# E5 run record: plan B deepened to 8 layers
Host node1. `launch/l8-prep.sh` (unit `launch/l8-prep.unit`, 2026-10-19 19:41Z) ran the resume check (`config/l8-resume-check.*`), fast_gates.py and the step-0 eval; `config/step8251-L8-copy.*` identify the expanded init.
Main command line: `launch/l8-codexmix.launch.sh`, unit `launch/l8-codexmix.unit` (started 2026-10-19 20:01Z); `launch/l8-codexmix.pipeline-start.txt` gives the layout (trainer GPUs 0-2, capture GPUs 3-5, ports 30931-30933).
`config/step-25562.manifest.json` identifies the final checkpoint on NFS.
Not recorded: the expand_layers.py command line (the initialization manifest gives its source, layer order and converter sha256) and the main run's launch.json.
