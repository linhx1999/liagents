from liagents.rl import RLTrainer
import json

# 创建训练工具
rl_trainer = RLTrainer("/home/linhx/models/Qwen2.5-0.5B-Instruct")

# 加载数据集
result = rl_trainer.load_dataset("/home/linhx/codebase/liagents/examples/datasets/gsm8k")
print(str(result))

# 评估模型
result = rl_trainer.evaluate(max_samples=4)
print(str(result))

print(f"\n评估完成!")
