#!/bin/bash
# Entry point for every script in this tree: `source env/env.sh` from the tree's root.
# ARC_ROOT is this tree; machine-specific locations come from env/local.sh, which you create
# from env/local.sh.example. Python reads them through src/arcforge/paths.py, which raises on
# first use of a variable that is not set.

export ARC_ROOT="${ARC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ARC_DATA="${ARC_DATA:-$ARC_ROOT/data}"

if [ -f "$ARC_ROOT/env/local.sh" ]; then
    source "$ARC_ROOT/env/local.sh"
fi
