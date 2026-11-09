"""TIR（tool-integrated reasoning）協定的單一實作。

這裡放的四樣東西，先前分別散在 `work/deepseek_api_bench/`（sandbox）、b200 的
`model_baseline/`（parsing）與 `nvarc_tir/hf_upload/config.json`（system prompt 與
tool schema），而 `work/tir_sft/eval_tir_qwen.py` 與 `work/expert_prune/eval_tir_prune.py`
各自複製了一份 prompt 渲染。學生模型是照 **teacher 生成訓練資料時那一份**協定訓練的，
所以協定只要有兩份就遲早會漂掉——評測看起來正常、分數卻在回答另一個問題。

`tir_config.json` 是 teacher rollout 當時的凍結設定，**不要改**：改了就不再是模型受訓的格式。
"""

__all__ = ["protocol", "parsing", "sandbox", "driver"]

