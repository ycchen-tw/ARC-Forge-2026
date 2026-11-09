"""Shared full-SFT trainer for the Qwen3.5 / Qwen3.6 family.

Promoted here from ``work/tir_sft/`` when a second unit (``work/qwen36_sft/``)
needed it -- CONVENTIONS §Directories: two consumers is the threshold, and units are
forbidden from importing each other.

``work/tir_sft/train_sft_opt.py`` and ``work/tir_sft/packed_text.py`` stay where
they are as frozen provenance for the 2026-08-22 runs. They are not maintained;
this is the copy that moves forward.

Run it as a module so the package resolves without path games::

    torchrun --nproc_per_node=4 -m arcforge.train.sft --model ... --data ...
"""
