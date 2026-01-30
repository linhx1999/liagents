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
    output_dir: str
    tokenizer: AutoTokenizer

    dataset: Optional[Any] = None

    # 训练配置
    num_epochs: int = 2
    learning_rate: float = 5e-5
    batch_size: int = 4

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

        return {
            "status": "success",
            "format_type": format_type,
            "split": split,
            "dataset_size": len(self.dataset),
            "sample_examples": self.dataset[:3],
        }

    def train(
        self,
        algorithm: Literal["sft", "grpo"] = "sft",
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
        use_tensorboard: bool = True,
    ) -> str:
        # 设置类参数
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        model_name = self.model_name_or_path.split("/")[-1].replace(" ", "_")

        print(f"\n{'='*60}\n")
        print(f"开始 {algorithm} 训练，模型: {model_name}")
        if custom_dataset is not None:
            print(f"数据集: 自定义数据集")
        else:
            print(f"数据集: {dataset_name or '已注册的数据集'}")
        print(f"训练轮数: {self.num_epochs}")
        print(f"输出目录: {self.output_dir}")
        print(f"算法: {algorithm}")
        if custom_reward is not None:
            print(f"奖励函数: 自定义奖励函数")

        if use_tensorboard:
            print(f"训练监控: tensorboard")

        print(f"\n{'='*60}\n")

        if not check_trl_installation():
            return json.dumps(
                {"status": "error", "message": "TRL不可用"}, ensure_ascii=False, indent=2
            )

        if algorithm == "sft":
            result = self._train_sft(
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                use_fp16=use_fp16,
                use_bf16=use_bf16,
                use_tensorboard=use_tensorboard,
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
        #         use_tensorboard=use_tensorboard
        #     )
        else:
            result = {
                "status": "error",
                "message": f"不支持的算法: {algorithm}。支持的算法: sft, grpo",
            }

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _train_sft(
        self,
        use_lora: bool,
        lora_rank: int,
        lora_alpha: int,
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_tensorboard: bool = True,
    ) -> dict[str, Any]:
        """执行SFT训练"""
        from trl import SFTConfig, SFTTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 创建监控配置
        report_to = ["tensorboard"] if use_tensorboard else ["none"]

        config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            fp16=use_fp16,
            bf16=use_bf16,
            report_to=report_to,
        )

        # 设置环境
        setup_training_environment(
            output_dir=self.output_dir,
            seed=42,
        )

        # 加载模型和 tokenizer
        print(f"加载模型: {self.model_name_or_path}")

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型
        device_map = "auto" if use_fp16 else None
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            device_map=device_map,
        )

        # 应用 LoRA（如果需要）
        if use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            print(f"LoRA 已应用 (rank={lora_rank}, alpha={lora_alpha})")

        print("模型加载完成")

        # 计算总步数
        total_steps = (
            len(self.dataset)
            // (config.per_device_train_batch_size * config.gradient_accumulation_steps)
        ) * config.num_train_epochs

        # 创建训练器
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=self.dataset,
            processing_class=tokenizer,
        )

        # 开始训练
        print("\n开始SFT训练...")
        trainer.train()
        print(f"\n{'='*80}")
        print("SFT训练完成")

        # 保存模型
        trainer.save_model()
        print(f"模型已保存到: {self.output_dir}")

        return {
            "status": "success",
            "algorithm": "SFT",
            "model": self.model_name_or_path,
            "output_dir": self.output_dir,
            "num_epochs": self.num_epochs,
            "dataset_size": len(self.dataset),
        }
