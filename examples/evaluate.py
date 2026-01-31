from liagents.rl import RLTrainer
import json

# 创建训练工具
rl_trainer = RLTrainer("/home/linhx/models/Qwen2.5-0.5B")

# 加载数据集
result = rl_trainer.load_dataset("/home/linhx/codebase/liagents/examples/datasets/gsm8k")
print(json.dumps(result, indent=2))

# 评估模型
result = rl_trainer.evaluate(max_samples=4)
print(json.dumps(result, indent=2))

print(f"\n评估完成!")
