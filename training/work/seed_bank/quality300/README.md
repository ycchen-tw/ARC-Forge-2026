# ARC2 quality-first source collection

This source package contains exactly 300 accepted tasks.

Open index.html for the offline gallery. It starts with task IDs, teaching pairs and held-out inputs; names, rules and held-out answers are revealed only on request.

Run `python -B tools/replay.py` with Python 3.12. It checks exact shown outputs, deterministic generation, nonmutation, latent oracles where supplied, hashes and the local import closure. It uses fixed mechanical-check seeds and makes no model calls. Core runtime uses the Python standard library.

Each `packages/q.../` includes `transform.py`, `generate_input.py`, `task.json`, `DOMAIN.md`, sanitized `seed.json`, `design_origin.json`, `provenance.json`, and an acceptance summary. Copy the whole task directory so any required local Python imports stay together.

- Traditional Chinese entry point
- Full production method and limitations (Traditional Chinese)
- [Engineering and replay scope](PIPELINE.md)
- Acceptance counts
- Omitted raw evidence and verified archive hashes
- Source license notices

The source snapshot is `55a85ef8a3e782419ac6d2c0c1c1b24474b194ae`. Every task has at least two distinct parents, including an eligible evaluation parent; some use three or more. `new_design` and `quality_rework`, and recorded `source_code_review` versus `summary_and_mechanism_comparison`, remain separately counted. These are synthetic compositions, not official benchmark tasks or claims of global novelty.

This lean package replays its included code and grids. It does not contain the raw solver contexts, original source snapshots, historical candidate trees or full acceptance harness, so it cannot independently rerun the archived acceptance decision.
