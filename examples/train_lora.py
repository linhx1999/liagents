from liagents.rl.trainning import RLTrainer

# 创建训练工具
rl_trainer = RLTrainer(
    model_name_or_path="/home/linhx/models/Qwen3-0.6B",
)

rl_trainer.load_dataset(
    dataset_name_or_path="/home/linhx/codebase/liagents/examples/datasets/gsm8k",
    format_type="sft",
)

# SFT训练 - 设置 use_fp16=False 禁用混合精度以避免兼容性问题
result = rl_trainer.train(
    algorithm="sft",
    learning_rate=5e-5,
    use_fp16=False,  # 禁用混合精度
)

print(f"\n✓ 训练完成!")
