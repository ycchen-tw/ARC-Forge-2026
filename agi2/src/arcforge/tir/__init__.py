"""The single implementation of the TIR (tool-integrated reasoning) protocol.

The four things here used to be scattered across `work/deepseek_api_bench/` (sandbox), the B200 node's `model_baseline/`
(parsing) and `nvarc_tir/hf_upload/config.json` (system prompt and tool schema), while
`work/tir_sft/eval_tir_qwen.py` and `work/expert_prune/eval_tir_prune.py` each copied the prompt rendering. The student
model was trained on **the exact protocol the teacher used to generate the training data**, so as soon as there are two
copies of the protocol they will drift sooner or later -- the eval looks normal while the score answers a different
question.

`tir_config.json` is the frozen config from the teacher rollouts; **do not change it**: once changed it is no longer
the format the model was trained on.
"""

__all__ = ["protocol", "parsing", "sandbox", "driver"]
