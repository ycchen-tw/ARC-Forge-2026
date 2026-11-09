# E4 run record: plan B (plain residual + KV injection)
Host node1. Conversion: `config/step9425-plain-inject.config.json` and `.initialization_manifest.json` identify the converted init (2026-10-18 16:15Z).
Main command line: `launch/plain-inject-codex1015.launch.sh`, unit `launch/plain-inject-codex1015.unit` (started 2026-10-18 17:14Z); `config/plain-inject-codex1015.launch.json` (trainer GPUs 0-2, capture GPUs 3-5) and `.run-config.json` are its resolved configs.
`launch/plain-resume-check{,2}.launch.sh` and their units are the two resume checks run before it.
Not recorded: the convert_plain.py command line (the initialization manifest gives its source, kind and converter sha256).
