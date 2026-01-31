from liagents.rl import RLTrainer

# 创建训练工具
rl_trainer = RLTrainer("Qwen/Qwen2.5-0.5B-Instruct")

# 加载数据集
result = rl_trainer.load_dataset("openai/gsm8k")
print(str(result))

# 评估模型
result = rl_trainer.evaluate(max_samples=100)
print(str(result))

print(f"\n评估完成!")
