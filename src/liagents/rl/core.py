"""RL训练核心功能模块

包含SFT和GRPO训练算法的具体实现。
"""

from typing import Dict, Any, Optional
from datasets import Dataset
from transformers import TrainerCallback
from dataclasses import dataclass, field

from .utils import check_trl_installation


@dataclass
class TrainingConfig:
    """训练配置类"""

    # 模型配置
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    model_revision: Optional[str] = None

    # 训练配置
    output_dir: str = "./output"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # RL特定配置
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # 硬件配置
    use_fp16: bool = True
    use_bf16: bool = False
    gradient_checkpointing: bool = True

    # LoRA配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])

    # 监控配置
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    use_tensorboard: bool = True

    # 其他配置
    seed: int = 42
    max_length: int = 2048

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class DetailedLoggingCallback(TrainerCallback):
    """详细日志回调

    在训练过程中输出详细的进度信息。
    """

    def __init__(self, total_steps: int, num_epochs: int):
        """初始化日志回调

        Args:
            total_steps: 总步数
            num_epochs: 训练轮数
        """
        self.total_steps = total_steps
        self.num_epochs = num_epochs
        self.current_epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        """每个epoch开始时调用"""
        self.current_epoch = state.epoch
        print(f"\n正在进行 {int(self.current_epoch) + 1}/{self.num_epochs} 轮（epoch）训练")

    def on_step_end(self, args, state, control, **kwargs):
        """每个step结束时调用"""
        if state.global_step % args.logging_steps == 0:
            loss = kwargs.get("loss", 0.0)
            print(
                f"   Step {state.global_step}/{self.total_steps} | "
                f"Loss: {loss:.4f} | "
                f"Learning Rate: {args.learning_rate:.2e}"
            )


class BaseTrainerWrapper:
    """训练器基类"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        初始化训练器

        Args:
            config: 训练配置
        """
        # 检查TRL是否安装
        if not check_trl_installation():
            raise ImportError("TRL 库未安装")

        self.config = config or TrainingConfig()
        self.trainer = None
        self.model = None
        self.tokenizer = None

    def setup_model(self):
        """设置模型和tokenizer"""
        raise NotImplementedError

    def train(self):
        """开始训练"""
        raise NotImplementedError

    def save_model(self, output_dir: Optional[str] = None):
        """
        保存模型

        Args:
            output_dir: 输出目录
        """
        save_dir = output_dir or self.config.output_dir
        if self.trainer:
            self.trainer.save_model(save_dir)
            print(f"模型已保存到: {save_dir}")
        else:
            print("训练器未初始化，无法保存模型")


class SFTTrainerWrapper(BaseTrainerWrapper):
    """SFT (Supervised Fine-Tuning) 训练器封装

    用于监督微调，让模型学会遵循指令和基本的推理格式。
    """

    def __init__(
        self, config: Optional[TrainingConfig] = None, dataset: Optional[Dataset] = None
    ):
        """初始化SFT训练器

        Args:
            config: 训练配置
            dataset: 训练数据集
        """
        super().__init__(config)
        self.dataset = dataset

    def train(self):
        """开始SFT训练"""
        from trl import SFTConfig, SFTTrainer

        if self.model is None:
            self.setup_model()

        if self.dataset is None:
            raise ValueError("数据集未设置，请提供训练数据集")

        # 确定report_to参数
        report_to = []
        if self.config.use_wandb:
            report_to.append("wandb")
        if self.config.use_tensorboard:
            report_to.append("tensorboard")
        if not report_to:
            report_to = ["none"]

        # 混合精度配置 - 使用用户配置
        use_fp16 = self.config.use_fp16
        use_bf16 = self.config.use_bf16

        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_length=self.config.max_length,
            report_to=report_to,
        )

        # 计算总步数
        total_steps = (
            len(self.dataset)
            // (
                self.config.per_device_train_batch_size
                * self.config.gradient_accumulation_steps
            )
        ) * self.config.num_train_epochs

        # 创建详细日志回调
        logging_callback = DetailedLoggingCallback(
            total_steps=total_steps, num_epochs=self.config.num_train_epochs
        )

        # 创建训练器
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
            callbacks=[logging_callback],
        )

        print("\n开始SFT训练...")
        print(f"{'='*80}\n")
        self.trainer.train()
        print(f"\n{'='*80}")
        print("SFT训练完成")

        return self.trainer

    def setup_model(self):
        """设置模型和tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"加载模型: {self.config.model_name_or_path}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型 - 使用用户配置决定是否使用混合精度
        device_map = "auto" if self.config.use_fp16 else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            device_map=device_map,
        )

        print("模型加载完成")


def setup_training_environment(config: TrainingConfig) -> None:
    """
    设置训练环境

    Args:
        config: 训练配置
    """
    import os
    import random
    import numpy as np

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 设置随机种子
    try:
        import torch

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    except ImportError:
        pass

    random.seed(config.seed)
    np.random.seed(config.seed)

    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 设置wandb配置
    if config.use_wandb:
        if config.wandb_project:
            os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_LOG_MODEL"] = "false"  # 不上传模型文件

    print(f"训练环境设置完成")
    print(f"   - 输出目录: {config.output_dir}")
    print(f"   - 随机种子: {config.seed}")
    print(f"   - 模型: {config.model_name_or_path}")
