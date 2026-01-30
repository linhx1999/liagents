from liagents.rl import RLTrainer

# 创建训练工具
rl_trainer = RLTrainer("/home/linhx/models/Qwen3-0.6B")

rl_trainer.load_dataset("/home/linhx/codebase/liagents/examples/datasets/gsm8k")

# SFT训练
result = rl_trainer.train()

print(f"\n训练完成!")
