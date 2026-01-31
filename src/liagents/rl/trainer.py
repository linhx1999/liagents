"""RL训练

提供强化学习训练功能，包括SFT、GRPO、PPO等算法。
现在采用模块化设计，将不同功能分离到独立的组件中。
"""

from typing import Any, Literal, Optional
from transformers import AutoTokenizer
import json
from datetime import datetime
from pathlib import Path

from .utils import (
    check_trl_installation,
    setup_training_environment,
)
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

    # 模型和输出配置
    model_name_or_path: str
    output_dir: str
    tokenizer: AutoTokenizer
    trained_model_path: Optional[str] = None

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
        is_think: bool = False,
    ) -> dict[str, Any]:
        """加载数据集

        Args:
            dataset_name_or_path: 数据集名称或路径
            format_type: 数据格式类型 ("sft" 或 "rl")
            split: 数据集分割 ("train" 或 "test")
            max_samples: 最大样本数，-1 表示全量使用
            is_think: 是否构建思维链，默认 False

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
                is_think=is_think,
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
        optimizer: str = "adamw_torch",
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: list[str] = ["q_proj", "v_proj"],
        lora_dropout: float = 0.05,
        lora_bias: Literal["none", "all", "lora_only"] = "none",
        lora_task_type: str = "CAUSAL_LM",
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
                optimizer=optimizer,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                lora_bias=lora_bias,
                lora_task_type=lora_task_type,
                use_fp16=use_fp16,
                use_bf16=use_bf16,
                use_tensorboard=use_tensorboard,
            )
            # 保存训练后的模型路径
            if result.get("status") == "success":
                self.trained_model_path = result.get("output_dir")
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
        optimizer: str = "adamw_torch",
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: list[str] = ["q_proj", "v_proj"],
        lora_dropout: float = 0.05,
        lora_bias: Literal["none", "all", "lora_only"] = "none",
        lora_task_type: str = "CAUSAL_LM",
        use_fp16: bool = False,
        use_bf16: bool = False,
        use_tensorboard: bool = True,
    ) -> dict[str, Any]:
        """执行SFT训练"""
        from trl import SFTConfig, SFTTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 创建监控配置
        report_to = ["tensorboard"] if use_tensorboard else ["none"]

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
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias=lora_bias,
                task_type=lora_task_type,
            )
            model = get_peft_model(model, lora_config)
            print(
                f"LoRA 已应用 (rank={lora_rank}, alpha={lora_alpha}, target_modules={lora_target_modules})"
            )

        print("模型加载完成")

        config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            optim=optimizer,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            fp16=use_fp16,
            bf16=use_bf16,
            report_to=report_to,
        )

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

    def evaluate(
        self,
        max_samples: int = -1,
        split: str = "test",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """评估模型性能

        功能：
        1. 加载测试数据集
        2. 加载模型和 tokenizer
        3. 生成预测
        4. 计算奖励/准确率
        5. 返回评估结果
        6. 保存评估结果到文件（使用类初始化时的 output_dir）

        Args:
            max_samples: 最大评估样本数（-1 表示使用全量数据集）
            split: 数据集分割（默认 "test"）
            max_new_tokens: 最大生成 token 数（默认 512）
            temperature: 采样温度（默认 0.7）
            do_sample: 是否采样（默认 False，使用贪婪解码）

        Returns:
            JSON 格式的评估结果
        """
        try:
            from .rewards import create_accuracy_reward
            from transformers import AutoModelForCausalLM
            import torch

            # 确定要使用的模型路径
            model_path = (
                self.trained_model_path
                if self.trained_model_path
                else self.model_name_or_path
            )
            model_source = "训练后的模型" if self.trained_model_path else "原始模型"

            # 加载测试数据
            print(f"加载测试数据集 (split={split})...")
            dataset = create_dataset(
                dataset_name_or_path="openai/gsm8k",
                format_type="rl",
                split=split,
                max_samples=max_samples,
                tokenizer=self.tokenizer,
            )
            print(f"已加载 {len(dataset)} 条数据")

            print(f"使用{model_source}: {model_path}...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, trust_remote_code=True
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                model.eval()
            except Exception as e:
                return json.dumps(
                    {"status": "error", "message": f"模型加载失败: {str(e)}"},
                    ensure_ascii=False,
                    indent=2,
                )

            # 生成预测
            print("生成预测...")
            completions = []
            ground_truths = []
            prompts = []

            # 创建迭代器
            from tqdm import tqdm

            iterator = tqdm(range(len(dataset)), desc="  评估进度", unit="样本")
            for i in iterator:
                prompt = dataset[i]["prompt"]
                ground_truth = dataset[i]["ground_truth"]
                prompts.append(prompt)

                # 生成回答
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                # 只取生成的部分，不包括输入
                completion = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )

                completions.append(completion)
                ground_truths.append(ground_truth)

            # 计算奖励
            print("计算评估指标...")
            reward_fn = create_accuracy_reward()
            rewards = reward_fn(completions, ground_truth=ground_truths)

            # 计算统计信息
            avg_reward = sum(rewards) / len(rewards)
            accuracy = avg_reward  # 对于准确性奖励，平均奖励就是准确率

            # 构建详细结果
            detailed_results = []
            for i, (prompt, completion, ground_truth, reward) in enumerate(
                zip(prompts, completions, ground_truths, rewards)
            ):
                detailed_results.append(
                    {
                        "index": i,
                        "prompt": prompt,
                        "completion": completion,
                        "ground_truth": ground_truth,
                        "reward": reward,
                        "correct": bool(reward),
                    }
                )

            result = {
                "status": "success",
                "model_path": model_path,
                "model_source": model_source,
                "num_samples": len(completions),
                "accuracy": f"{accuracy:.2%}",
                "average_reward": f"{avg_reward:.4f}",
                "device": device,
                "eval_config": {
                    "split": split,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": do_sample,
                },
                "detailed_results": detailed_results,
            }

            print(f"\n评估完成!")
            print(f"  准确率: {accuracy:.2%}")
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  正确样本: {sum(rewards)}/{len(rewards)}")

            # 保存评估结果
            eval_dir = Path(self.output_dir) / f"evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)

            results_file = eval_dir / "evaluation_results.json"

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"  结果已保存至: {results_file}")

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps(
                {"status": "error", "message": f"评估失败: {str(e)}"},
                ensure_ascii=False,
                indent=2,
            )
