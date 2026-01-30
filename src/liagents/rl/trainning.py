"""RL训练

提供强化学习训练功能，包括SFT、GRPO、PPO等算法。
现在采用模块化设计，将不同功能分离到独立的组件中。
"""

from typing import Any, Literal
from transformers import AutoTokenizer
import json

from .core import RLTrainingCore
from .handler.reward_handler import RLRewardHandler
from .handler.evaluation_handler import RLEvaluationHandler
from .datasets import create_dataset


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

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", use_lora: bool = True, output_dir: str = "./outputs"):
        self.model_name = model_name
        self.use_lora = use_lora
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.training_core = RLTrainingCore()
        self.reward_handler = RLRewardHandler()
        self.evaluation_handler = RLEvaluationHandler()
        self.custom_datasets = {}

    def train(
        self,
        algorithm: str = "sft",
        model_name: str = "Qwen/Qwen3-0.6B",
        dataset_name: str = "gsm8k",
        num_epochs: int = 2,
        output_dir: str = "./outputs",
        use_lora: bool = True,
        batch_size: int = 4,
        max_samples: int | None = None,
        custom_dataset: Any = None,
        custom_reward: Any = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: str | None = None,
    ) -> str:
        """训练模型

        Args:
            algorithm: 训练算法 (sft/grpo)
            model_name: 模型名称
            dataset_name: 数据集名称
            num_epochs: 训练轮数
            output_dir: 输出目录
            use_lora: 是否使用LoRA
            batch_size: 批次大小
            max_samples: 最大样本数
            custom_dataset: 自定义数据集
            custom_reward: 自定义奖励函数
            use_wandb: 使用 wandb 监控
            use_tensorboard: 使用 tensorboard 监控
            wandb_project: wandb 项目名称
        """
        algorithm = algorithm.lower().strip()

        print(f"\n{'='*60}\n")
        print(f"开始 {algorithm.upper()} 训练")
        print(f"模型: {model_name}")
        if custom_dataset:
            print(f"数据集: 自定义数据集 ")
        else:
            print(f"数据集: {dataset_name}")
        print(f"训练轮数: {num_epochs}")
        print(f"输出目录: {output_dir}")
        print(f"算法: {algorithm.upper()}")
        if custom_reward:
            print(f"奖励函数: 自定义奖励函数")

        monitoring = []
        if use_wandb:
            monitoring.append(f"wandb (项目: {wandb_project or 'default'})")
        if use_tensorboard:
            monitoring.append("tensorboard")
        if monitoring:
            print(f"训练监控: {', '.join(monitoring)}")

        print(f"\n{'='*60}\n")

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

    def load_dataset(
        self,
        dataset_name_or_path: str = "openai/gsm8k",
        format_type: Literal["sft", "rl"] = "sft",
        split: str = "train",
        max_samples: int = 100,
    ) -> str:

        format_type = format_type.lower().strip()
        if format_type in ["sft", "rl"]:
            dataset = create_dataset(
                dataset_name_or_path=dataset_name_or_path,
                format_type=format_type,
                max_samples=max_samples,
                split=split,
                tokenizer=self.tokenizer
            )
        else:
            return json.dumps({
                "status": "error",
                "message": f"不支持的数据格式: {format_type}。支持的格式: sft, rl"
            }, ensure_ascii=False, indent=2)

        result = {
            "status": "success",
            "format_type": format_type,
            "split": split,
            "dataset_size": len(dataset),
            "sample_examples": dataset[:3] if len(dataset) > 3 else []
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
