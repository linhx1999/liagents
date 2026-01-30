"""RL训练

提供强化学习训练功能，包括SFT、GRPO、PPO等算法。
现在采用模块化设计，将不同功能分离到独立的组件中。
"""

from typing import Any, Literal, Optional
from transformers import AutoTokenizer
import json
from datetime import datetime
from pathlib import Path

from .core import (
    TrainingConfig,
    SFTTrainerWrapper,
    setup_training_environment,
)
from .handler.reward_handler import RLRewardHandler
from .handler.evaluation_handler import RLEvaluationHandler
from .datasets import create_dataset
from .utils import check_trl_installation


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

    # 模型和输出配置
    model_name_or_path: str
    tokenizer: Any = None
    output_dir: str

    # 训练配置
    algorithm: str = "sft"
    num_epochs: int = 2
    learning_rate: float = 5e-5
    batch_size: int = 4
    max_samples: int = -1  # -1 表示全量使用数据集

    # LoRA配置
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16

    # 精度配置
    use_fp16: bool = False
    use_bf16: bool = True

    # 数据集配置
    dataset: Optional[Any] = None

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-0.6B",
        output_dir: str = "./outputs",
    ):
        """初始化训练器

        Args:
            model_name_or_path: 模型名称或路径
            output_dir: 输出目录基础路径
        """
        # 模型配置
        self.model_name_or_path = model_name_or_path

        # 输出配置
        # 创建带模型名和时间戳的输出目录
        model_name = model_name_or_path.split("/")[-1].replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = str(Path(output_dir) / model_name / timestamp)

        # 初始化 tokenizer 和 handlers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # self.reward_handler = RLRewardHandler()
        # self.evaluation_handler = RLEvaluationHandler()

    def train(
        self,
        algorithm: str = "sft",
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        batch_size: int = 4,
        num_epochs: int = 2,
        learning_rate: float = 5e-5,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        use_fp16: bool = False,
        use_bf16: bool = False,
        custom_dataset: Optional[Any] = None,
        custom_reward: Optional[Any] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: Optional[str] = None,
    ) -> str:
        """训练模型

        Args:
            algorithm: 训练算法 (sft/grpo)，默认为 "sft"
            model_name: 模型名称，默认使用初始化时的值
            dataset_name: 数据集名称，默认使用初始化时的值
            max_samples: 最大样本数，-1 表示全量使用
            batch_size: 批次大小，默认为 4
            num_epochs: 训练轮数，默认为 2
            learning_rate: 学习率，默认为 5e-5
            use_lora: 是否使用LoRA，默认为 True
            lora_rank: LoRA rank，默认为 8
            lora_alpha: LoRA alpha，默认为 16
            use_fp16: 是否使用 FP16 混合精度，默认为 False
            use_bf16: 是否使用 BF16 混合精度，默认为 False
            custom_dataset: 自定义数据集
            custom_reward: 自定义奖励函数
            use_wandb: 使用 wandb 监控，默认为 False
            use_tensorboard: 使用 tensorboard 监控，默认为 True
            wandb_project: wandb 项目名称
        """
        # 使用传入的参数，如果没有则使用类属性
        algorithm = algorithm.lower().strip()
        model_name = model_name or self.model_name_or_path

        print(f"\n{'='*60}\n")
        print(f"开始 {algorithm.upper()} 训练")
        print(f"模型: {model_name}")
        if custom_dataset is not None:
            print(f"数据集: 自定义数据集")
        else:
            print(f"数据集: {dataset_name or '已注册的数据集'}")
        print(f"训练轮数: {num_epochs}")
        print(f"输出目录: {self.output_dir}")
        print(f"算法: {algorithm.upper()}")
        if custom_reward is not None:
            print(f"奖励函数: 自定义奖励函数")

        monitoring = []
        if use_wandb:
            monitoring.append(f"wandb (项目: {wandb_project or 'default'})")
        if use_tensorboard:
            monitoring.append("tensorboard")
        if monitoring:
            print(f"训练监控: {', '.join(monitoring)}")

        print(f"\n{'='*60}\n")

        if not check_trl_installation():
            return json.dumps(
                {"status": "error", "message": "TRL不可用"}, ensure_ascii=False, indent=2
            )

        if algorithm == "sft":
            result = self._train_sft(
                num_epochs=num_epochs,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                batch_size=batch_size,
                use_fp16=use_fp16,
                use_bf16=use_bf16,
                custom_dataset=custom_dataset,
                use_wandb=use_wandb,
                use_tensorboard=use_tensorboard,
                wandb_project=wandb_project,
            )
        # elif algorithm == "grpo":
        #     result = self.training_core.train_grpo(
        #         model_name=model_name,
        #         dataset_name=dataset_name,
        #         max_samples=max_samples,
        #         num_epochs=num_epochs,
        #         output_dir=output_dir,
        #         use_lora=use_lora,
        #         batch_size=batch_size,
        #         custom_dataset=custom_dataset,
        #         custom_reward=custom_reward,
        #         use_wandb=use_wandb,
        #         use_tensorboard=use_tensorboard,
        #         wandb_project=wandb_project
        #     )
        else:
            result = {
                "status": "error",
                "message": f"不支持的算法: {algorithm}。支持的算法: sft, grpo",
            }

        return json.dumps(result, ensure_ascii=False, indent=2)

    def load_dataset(
        self,
        dataset_name_or_path: str = "openai/gsm8k",
        format_type: Literal["sft", "rl"] = "sft",
        split: str = "train",
        max_samples: int = -1,
    ) -> dict[str, Any]:
        """加载数据集

        Args:
            dataset_name_or_path: 数据集名称或路径
            format_type: 数据格式类型 ("sft" 或 "rl")
            split: 数据集分割 ("train" 或 "test")
            max_samples: 最大样本数，-1 表示全量使用

        Returns:
            包含加载结果信息的字典
        """
        if format_type in ("sft", "rl"):
            self.dataset = create_dataset(
                dataset_name_or_path=dataset_name_or_path,
                format_type=format_type,
                max_samples=max_samples,
                split=split,
                tokenizer=self.tokenizer,
            )
        else:
            return {
                "status": "error",
                "message": f"不支持的数据格式: {format_type}。支持的格式: sft, rl",
            }

        result = {
            "status": "success",
            "format_type": format_type,
            "split": split,
            "dataset_size": len(self.dataset),
            "sample_examples": self.dataset[:3] if len(self.dataset) > 3 else [],
        }
        return result

    def _train_sft(
        self,
        num_epochs: int,
        use_lora: bool,
        lora_rank: int,
        lora_alpha: int,
        batch_size: int,
        use_fp16: bool = False,
        use_bf16: bool = False,
        custom_dataset=None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: Optional[str] = None,
    ) -> dict[str, Any]:
        """执行SFT训练"""
        # 创建配置
        config = TrainingConfig(
            model_name_or_path=self.model_name_or_path,
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            use_lora=use_lora,
            lora_r=lora_rank,
            lora_alpha=lora_alpha,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            wandb_project=wandb_project,
        )

        # 设置环境
        setup_training_environment(config)

        # 加载数据集
        if custom_dataset is not None:
            # 使用自定义数据集
            dataset = custom_dataset
            print(f"使用自定义数据集: {len(dataset)} 个样本")
        elif self.dataset is not None:
            dataset = self.dataset
            print(f"使用注册的数据集: {len(dataset)} 个样本")
        else:
            raise ValueError("未指定数据集，请先加载数据集")

        # 创建训练器
        trainer_wrapper = SFTTrainerWrapper(config=config, dataset=dataset)

        # 开始训练
        trainer_wrapper.train()

        # 保存模型
        trainer_wrapper.save_model()

        return {
            "status": "success",
            "algorithm": "SFT",
            "model": self.model_name_or_path,
            "output_dir": self.output_dir,
            "num_epochs": num_epochs,
            "dataset_size": len(dataset),
        }
