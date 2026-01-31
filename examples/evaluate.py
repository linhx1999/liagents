from liagents.rl import RLTrainer
import json

# 创建训练工具
rl_trainer = RLTrainer("/root/autodl-tmp/outputs/Qwen2.5-0.5B-Instruct/20260131-160503")

# 加载数据集
result = rl_trainer.load_dataset("/root/autodl-tmp/datasets/gsm8k")
print(str(result))

# 评估模型
result = rl_trainer.evaluate(max_samples=100)
print(str(result))

print(f"\n评估完成!")
