"""RL训练

提供强化学习训练功能，包括SFT、GRPO、PPO等算法。
现在采用模块化设计，将不同功能分离到独立的组件中。
"""

from typing import Dict, Any
import json
from .core import RLTrainingCore
from .handler.data_handler import RLDataHandler
from .handler.reward_handler import RLRewardHandler
from .handler.evaluation_handler import RLEvaluationHandler


class RLTrainer:
    """RL训练工具 - 主要入口点

    支持的训练算法：
    - SFT: Supervised Fine-Tuning (监督微调)
    - GRPO: Group Relative Policy Optimization (群体相对策略优化)

    支持的功能：
    - 训练模型 (train)
    - 加载数据集 (load_dataset)
    - 创建奖励函数 (create_reward)
    - 评估模型 (evaluate)
    """

    def __init__(self):
        self.training_core = RLTrainingCore()
        self.data_handler = RLDataHandler()
        self.reward_handler = RLRewardHandler()
        self.evaluation_handler = RLEvaluationHandler()

    def train(self, parameters: Dict[str, Any]) -> str:
        """训练模型

        Args:
            parameters: 训练参数，包含:
                - algorithm: 训练算法 (sft/grpo)
                - model_name: 模型名称
                - dataset: 数据集名称
                - num_epochs: 训练轮数
                - output_dir: 输出目录
                - use_lora: 是否使用LoRA
                - batch_size: 批次大小
        """
        algorithm = parameters.get("algorithm", "sft").lower()
        model_name = parameters.get("model_name", "Qwen/Qwen2-0.5B-Instruct")
        dataset_name = parameters.get("dataset", "gsm8k")
        max_samples = parameters.get("max_samples", None)
        num_epochs = parameters.get("num_epochs", 3)
        output_dir = parameters.get("output_dir", "./output")
        use_lora = parameters.get("use_lora", True)
        batch_size = parameters.get("batch_size", 4)
        custom_dataset = parameters.get("custom_dataset", None)
        custom_reward = parameters.get("custom_reward", None)
        use_wandb = parameters.get("use_wandb", False)
        use_tensorboard = parameters.get("use_tensorboard", True)
        wandb_project = parameters.get("wandb_project", None)

        print(f"\n{'='*60}")
        print(f"🚀 开始 {algorithm.upper()} 训练")
        print(f"{'='*60}")
        print(f"📦 模型: {model_name}")
        if custom_dataset:
            print(f"📊 数据集: 自定义数据集")
        else:
            print(f"📊 数据集: {dataset_name}")
        print(f"🔄 训练轮数: {num_epochs}")
        print(f"💾 输出目录: {output_dir}")
        print(f"🎯 算法: {algorithm.upper()}")
        if custom_reward:
            print(f"🎁 奖励函数: 自定义奖励函数")

        monitoring = []
        if use_wandb:
            monitoring.append(f"wandb (项目: {wandb_project or 'default'})")
        if use_tensorboard:
            monitoring.append("tensorboard")
        if monitoring:
            print(f"📊 训练监控: {', '.join(monitoring)}")

        print(f"{'='*60}\n")

        if not self.training_core.trl_available:
            return json.dumps({
                "status": "error",
                "message": "TRL未安装"
            }, ensure_ascii=False, indent=2)

        if algorithm == "sft":
            result = self.training_core.train_sft(
                model_name=model_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                num_epochs=num_epochs,
                output_dir=output_dir,
                use_lora=use_lora,
                batch_size=batch_size,
                custom_dataset=custom_dataset,
                use_wandb=use_wandb,
                use_tensorboard=use_tensorboard,
                wandb_project=wandb_project
            )
        elif algorithm == "grpo":
            result = self.training_core.train_grpo(
                model_name=model_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                num_epochs=num_epochs,
                output_dir=output_dir,
                use_lora=use_lora,
                batch_size=batch_size,
                custom_dataset=custom_dataset,
                custom_reward=custom_reward,
                use_wandb=use_wandb,
                use_tensorboard=use_tensorboard,
                wandb_project=wandb_project
            )
        else:
            result = {
                "status": "error",
                "message": f"不支持的算法: {algorithm}。支持的算法: sft, grpo"
            }

        return json.dumps(result, ensure_ascii=False, indent=2)

    def load_dataset(self, parameters: Dict[str, Any]) -> str:
        """加载数据集"""
        return self.data_handler.handle_load_dataset(parameters)

    def create_reward(self, parameters: Dict[str, Any]) -> str:
        """创建奖励函数"""
        return self.reward_handler.handle_create_reward(parameters)

    def evaluate(self, parameters: Dict[str, Any]) -> str:
        """评估模型"""
        return self.evaluation_handler.handle_evaluate(parameters)

    # 便捷函数接口
    def register_dataset(self, name: str, dataset) -> None:
        """
        注册自定义数据集

        Args:
            name: 数据集名称
            dataset: 数据集对象(HuggingFace Dataset)
        """
        self.data_handler.register_dataset(name, dataset)

    def register_reward_function(self, name: str, reward_fn) -> None:
        """
        注册自定义奖励函数

        Args:
            name: 奖励函数名称
            reward_fn: 奖励函数(接受completions和kwargs,返回rewards列表)
        """
        self.reward_handler.register_reward_function(name, reward_fn)