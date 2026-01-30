from liagents.rl import RLTrainer
import json


# 创建工具
rl_trainer = RLTrainer(
    model_name_or_path="/home/linhx/models/Qwen3-0.6B",
)

# 1. 加载SFT格式数据集
sft_result = rl_trainer.load_dataset(
    dataset_name_or_path="/home/linhx/codebase/liagents/examples/datasets/gsm8k",
    format_type="sft",
    max_samples=5  # 只加载5个样本查看
)

print(json.dumps(sft_result, indent=2, ensure_ascii=False))

# 2. 加载RL格式数据集
rl_result = rl_trainer.load_dataset(
    dataset_name_or_path="/home/linhx/codebase/liagents/examples/datasets/gsm8k",
    format_type="rl",
    max_samples=5
)

print(json.dumps(rl_result, indent=2, ensure_ascii=False))
