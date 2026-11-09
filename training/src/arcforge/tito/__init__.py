"""arc-forge side of the TITO session ledger (``docs/plans/tito_session_ledger.md``).

- ``preflight`` / ``manifest``: harness side (stdlib only; arc2 loads them by file path).
- ``ledger``: read ``DIR/<instance>/`` → ``Segment``, check the invariants.
- ``roles``: role (``0 (1 2)* 1?``) and per-token span_kind.
- ``join``: ``tito_manifest.json`` → task / game and outcome; rid join.
- ``export``: SFT / DFlash / RL parquet.
- ``audit``: collection audit (``collection_passed``).
- ``python -m arcforge.tito``: CLI.

This file deliberately imports no submodule: ``import arcforge.tito.preflight`` must not drag in numpy / pyarrow.
"""
