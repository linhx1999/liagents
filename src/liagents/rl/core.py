"""RL训练核心功能模块

包含SFT和GRPO训练算法的具体实现。
"""

from typing import Dict, Any, Optional


class RLTrainingCore:
    """RL训练核心功能类，包含具体的训练实现"""

    trl_available: bool = False
    
    def __init__(self):
        try:
            import trl
            self.trl_available = True
        except ImportError:
            self.trl_available = False


    def train_sft(
        self,
        model_name: str,
        dataset_name: str,
        max_samples: Optional[int],
        num_epochs: int,
        output_dir: str,
        use_lora: bool,
        batch_size: int,
        custom_dataset = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: Optional[str] = None
    ) -> Dict[str, Any]:
        """执行SFT训练"""
        from hello_agents.rl import (
            SFTTrainerWrapper,
            TrainingConfig,
            create_sft_dataset,
            setup_training_environment
        )

        # 创建配置
        config = TrainingConfig(
            model_name=model_name,
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            use_lora=use_lora,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            wandb_project=wandb_project
        )

        # 设置环境
        setup_training_environment(config)

        # 加载数据集
        if custom_dataset is not None:
            # 使用自定义数据集
            dataset = custom_dataset
            print(f"✅ 使用自定义数据集: {len(dataset)} 个样本")
        else:
            # 使用默认数据集
            dataset = create_sft_dataset(max_samples=max_samples)

        # 创建训练器
        trainer_wrapper = SFTTrainerWrapper(config=config, dataset=dataset)

        # 开始训练
        trainer_wrapper.train()

        # 保存模型
        trainer_wrapper.save_model()

        return {
            "status": "success",
            "algorithm": "SFT",
            "model": model_name,
            "output_dir": output_dir,
            "num_epochs": num_epochs,
            "dataset_size": len(dataset)
        }

    def train_grpo(
        self,
        model_name: str,
        dataset_name: str,
        max_samples: Optional[int],
        num_epochs: int,
        output_dir: str,
        use_lora: bool,
        batch_size: int,
        custom_dataset = None,
        custom_reward = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: Optional[str] = None
    ) -> Dict[str, Any]:
        """执行GRPO训练"""
        from hello_agents.rl import (
            GRPOTrainerWrapper,
            TrainingConfig,
            create_rl_dataset,
            create_accuracy_reward,
            setup_training_environment
        )

        # 创建配置
        config = TrainingConfig(
            model_name=model_name,
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            use_lora=use_lora,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            wandb_project=wandb_project
        )

        # 设置环境
        setup_training_environment(config)

        # 加载数据集
        if custom_dataset is not None:
            # 使用自定义数据集
            dataset = custom_dataset
            print(f"✅ 使用自定义数据集: {len(dataset)} 个样本")
        else:
            # 使用默认数据集
            dataset = create_rl_dataset(max_samples=max_samples, model_name=model_name)

        # 创建奖励函数
        if custom_reward is not None:
            # 使用自定义奖励函数
            reward_fn = custom_reward
            print(f"✅ 使用自定义奖励函数")
        else:
            # 使用默认奖励函数
            reward_fn = create_accuracy_reward()

        # 创建训练器
        trainer_wrapper = GRPOTrainerWrapper(
            config=config,
            dataset=dataset,
            reward_fn=reward_fn
        )

        # 开始训练
        trainer_wrapper.train()

        # 保存模型
        trainer_wrapper.save_model()

        return {
            "status": "success",
            "algorithm": "GRPO",
            "model": model_name,
            "output_dir": output_dir,
            "num_epochs": num_epochs,
            "dataset_size": len(dataset)
        }