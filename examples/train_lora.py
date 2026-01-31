from liagents.rl import RLTrainer

# 创建训练工具
rl_trainer = RLTrainer("/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct", output_dir="/root/autodl-tmp/outputs")

# 加载数据集
rl_trainer.load_dataset("/root/autodl-tmp/datasets/gsm8k")

# SFT训练
result = rl_trainer.train()

print(f"\n训练完成!")
