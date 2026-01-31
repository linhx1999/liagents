from liagents.rl import RLTrainer

# 创建训练工具
rl_trainer = RLTrainer("/home/linhx/models/Qwen3-0.6B")

# 加载数据集
rl_trainer.load_dataset("/home/linhx/codebase/liagents/examples/datasets/gsm8k")

# 评估模型
result = rl_trainer.evaluate(max_samples=4)

print(f"\n评估完成!")
