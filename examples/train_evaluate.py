from liagents.rl import RLTrainer

# 创建训练工具
rl_trainer = RLTrainer("/home/linhx/models/Qwen2.5-0.5B-Instruct")

# 加载数据集
rl_trainer.load_dataset(
    "/home/linhx/codebase/liagents/examples/datasets/gsm8k", max_samples=32
)

# 训练
result = rl_trainer.train()

# 评估模型
result = rl_trainer.evaluate(max_samples=4)

print(f"\n训练完成!")
