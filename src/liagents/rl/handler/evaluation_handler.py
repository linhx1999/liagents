"""RL模型评估模块

包含模型评估相关的功能。
"""

from typing import Dict, Any
import json


class RLEvaluationHandler:
    """RL评估处理类，负责模型评估功能"""

    def handle_evaluate(self, parameters: Dict[str, Any]) -> str:
        """处理模型评估操作"""
        try:
            from hello_agents.rl import (
                create_rl_dataset,
                create_accuracy_reward,
                evaluate_rewards
            )
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_path = parameters.get("model_path")
            max_samples = parameters.get("max_samples", 100)

            if not model_path:
                return json.dumps({
                    "status": "error",
                    "message": "缺少必需参数: model_path"
                }, ensure_ascii=False, indent=2)

            # 加载测试数据
            print(f"📥 加载测试数据集 (max_samples={max_samples})...")
            dataset = create_rl_dataset(split="test", max_samples=max_samples, model_name=model_path)

            # 加载模型和tokenizer
            print(f"📥 加载模型: {model_path}...")
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                model.eval()
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"模型加载失败: {str(e)}"
                }, ensure_ascii=False, indent=2)

            # 生成预测
            print("🔮 生成预测...")
            completions = []
            ground_truths = []

            # 导入tqdm用于进度条
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False
                print("  提示: 安装tqdm可显示进度条 (pip install tqdm)")

            # 创建迭代器
            iterator = range(min(max_samples, len(dataset)))
            if use_tqdm:
                iterator = tqdm(iterator, desc="  评估进度", unit="样本")

            for i in iterator:
                prompt = dataset[i]["prompt"]
                ground_truth = dataset[i]["ground_truth"]

                # 生成回答
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,  # 减少生成长度加快速度
                        temperature=0.7,
                        do_sample=False,  # 使用贪婪解码加快速度
                        pad_token_id=tokenizer.pad_token_id
                    )
                # 只取生成的部分,不包括输入
                completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                completions.append(completion)
                ground_truths.append(ground_truth)

                # 如果没有tqdm,每10个样本打印一次进度
                if not use_tqdm and (i + 1) % 10 == 0:
                    print(f"  进度: {i+1}/{max_samples}")

            # 计算奖励
            print("📊 计算评估指标...")
            reward_fn = create_accuracy_reward()
            rewards = reward_fn(completions, ground_truth=ground_truths)

            # 计算统计信息
            avg_reward = sum(rewards) / len(rewards)
            accuracy = avg_reward  # 对于准确性奖励,平均奖励就是准确率

            result = {
                "status": "success",
                "model_path": model_path,
                "num_samples": len(completions),
                "accuracy": f"{accuracy:.2%}",
                "average_reward": f"{avg_reward:.4f}",
                "device": device
            }

            print(f"\n✅ 评估完成!")
            print(f"  准确率: {accuracy:.2%}")
            print(f"  平均奖励: {avg_reward:.4f}")

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"评估失败: {str(e)}"
            }, ensure_ascii=False, indent=2)