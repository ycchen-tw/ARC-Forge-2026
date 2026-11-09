# E3 run record: D-line v3wsd and its codex1015 extension
Host node1. v3wsd: `config/dline-v3wsd.launch.json` (trainer GPUs 0-3, capture GPUs 4-6, launched 2026-10-13 15:15Z) and `config/dline-v3wsd.run-config.json`; `config/dline-v3wsd.step-34404.manifest.json` identifies the parent checkpoint.
Extension 20261017-dline-ext-codex1015: started 2026-10-17 (trainer GPUs 0,1, capture GPUs 2,3, ports 30801/30802), resumed from step 6000 on 2026-10-18 07:46Z by `launch/dline-ext-codex1015.autoresume.sh` as unit `launch/dline-ext-codex1015.unit`. `launch/dline-ext-codex1015.resume-start.txt` gives its init, plan and gate; `launch/dline-ext-codex1015.pipeline-start.txt` its GPU layout.
`config/codex1015-resume-check.*` and `launch/codex1015-resume-check.unit` are the pre-run resume check; `config/dline-ext-codex1015.pre-decay.manifest.json` identifies the pre-decay checkpoint.
Not recorded: the extension's first launch command and launch.json; the capture fork trees are not in this tree.
