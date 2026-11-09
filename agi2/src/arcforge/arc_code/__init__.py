"""arc-code's shared core: gateway, ledger, tokens, convenience client, dependency manifest.

Design: ``docs/plans/arc_code_gateway.md``. All four arms share this; each one's
``work/arc_code/arms/<driver>/`` keeps only its adapter and mount list.
"""

__all__ = ["engine", "errors", "gateway", "ledger", "server", "tokens"]
