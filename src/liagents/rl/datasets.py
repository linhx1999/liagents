from typing import Dict, Any, Optional, Literal
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


class BaseDataset(ABC):
    """数据集基类

    提供数据集加载、样本限制和格式化的通用接口。
    子类需要实现 format_for_sft 和 format_for_rl 方法。
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        format_type: Literal["sft", "rl"] = "sft",
        tokenizer: Optional[AutoTokenizer] = None,
        subset: Optional[str] = None,
    ):
        """初始化数据集

        Args:
            dataset_name_or_path: 数据集名称或路径
            split: 数据集分割 ("train" 或 "test")
            max_samples: 最大样本数，-1 或 None 表示全量使用，>0 表示限制数量
            format_type: 数据格式类型 ("sft" 用于监督学习, "rl" 用于强化学习)
            tokenizer: Tokenizer对象,用于RL格式应用chat template
            subset: 数据集子集名称（某些数据集需要）
        """
        self.dataset_name_or_path = dataset_name_or_path
        self.split = split
        self.max_samples = max_samples
        self.format_type = format_type
        self.tokenizer = tokenizer
        self.subset = subset

        print(f"加载 {self.__class__.__name__} 数据集 (split={split})...")
        self.dataset = self._load_dataset(max_samples=max_samples)

    def _load_dataset(self, max_samples: Optional[int] = None) -> Dataset:
        """加载数据集

        Args:
            max_samples: 最大样本数，-1 或 None 表示全量使用，>0 表示限制数量

        Returns:
            HuggingFace Dataset对象（可能已被限制数量）
        """
        # 加载原始数据集
        if self.subset:
            dataset = load_dataset(
                self.dataset_name_or_path, self.subset, split=self.split
            )
        else:
            dataset = load_dataset(self.dataset_name_or_path, split=self.split)

        # 处理样本数量限制
        # max_samples: -1 或 None 表示全量，>0 表示限制数量
        if max_samples is not None and max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            print(f"   使用 {len(dataset)} 个样本（限制：{max_samples}）")
        else:
            # max_samples 为 None 或 -1 时，使用全量数据集
            print(f"   加载了 {len(dataset)} 个样本")

        return dataset

    @abstractmethod
    def format_for_sft(self, example: Dict[str, Any]) -> Dict[str, str]:
        """格式化为SFT训练格式

        Args:
            example: 原始数据样本

        Returns:
            格式化后的样本，包含 "prompt" 和 "completion"
        """
        pass

    @abstractmethod
    def format_for_rl(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """格式化为RL训练格式

        Args:
            example: 原始数据样本

        Returns:
            格式化后的样本，包含 "prompt" 和 "ground_truth" 等字段
        """
        pass

    def get_dataset(self) -> Dataset:
        """获取格式化后的数据集

        Returns:
            HuggingFace Dataset对象
        """
        if self.format_type == "sft":
            format_fn = self.format_for_sft
        elif self.format_type == "rl":
            format_fn = self.format_for_rl
        else:
            raise ValueError(f"不支持的格式类型: {self.format_type}")

        formatted_dataset = self.dataset.map(
            format_fn, remove_columns=self.dataset.column_names
        )

        return formatted_dataset

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        example = self.dataset[idx]
        if self.format_type == "sft":
            return self.format_for_sft(example)
        else:
            return self.format_for_rl(example)


def create_dataset(
    dataset_name_or_path: str = "openai/gsm8k",
    format_type: Literal["sft", "rl"] = "sft",
    max_samples: Optional[int] = -1,
    split: str = "train",
    tokenizer: Optional[AutoTokenizer] = None,
    is_think: bool = False,
) -> Dataset:
    """创建数据集

    Args:
        dataset_name_or_path: 数据集名称或路径
        format_type: 数据格式 (sft/rl)
        max_samples: 最大样本数，-1 或 None 表示全量使用
        split: 数据集分割
        tokenizer: Tokenizer对象
        is_think: 是否构建思维链

    Returns:
        格式化后的数据集
    """
    if "gsm8k" in dataset_name_or_path.lower().strip():
        dataset_wrapper = GSM8KDataset(
            dataset_name_or_path=dataset_name_or_path,
            split=split,
            max_samples=max_samples,
            format_type=format_type,
            tokenizer=tokenizer,
            is_think=is_think,
        )
        return dataset_wrapper.get_dataset()
    else:
        raise ValueError(f"不支持的数据集: {dataset_name_or_path}")


class GSM8KDataset(BaseDataset):
    """GSM8K数学推理数据集

    GSM8K (Grade School Math 8K) 是一个包含8500个高质量小学数学问题的数据集。
    每个问题都需要2-8步的推理过程来解决。
    """

    def __init__(
        self,
        dataset_name_or_path: str = "openai/gsm8k",
        split: str = "train",
        max_samples: int = -1,
        format_type: Literal["sft", "rl"] = "sft",
        tokenizer: Optional[AutoTokenizer] = None,  # 用于RL格式应用chat template
        is_think: bool = False,
    ):
        """
        初始化GSM8K数据集

        Args:
            dataset_name_or_path: 数据集名称或路径
            split: 数据集分割 ("train" 或 "test")
            max_samples: 最大样本数，-1 或 None 表示全量使用
            format_type: 数据格式类型 ("sft" 用于监督学习, "rl" 用于强化学习)
            tokenizer: Tokenizer对象,用于RL格式应用chat template
            is_think: 是否构建思维链
        """
        self.is_think = is_think
        super().__init__(
            dataset_name_or_path=dataset_name_or_path,
            split=split,
            max_samples=max_samples,
            format_type=format_type,
            tokenizer=tokenizer,
            subset="main",  # GSM8K 需要指定 subset
        )

    def format_for_sft(self, example: Dict[str, Any]) -> Dict[str, str]:
        """格式化为SFT训练格式

        Args:
            example: 原始数据样本

        Returns:
            格式化后的样本，包含 "prompt" 和 "completion"
        """
        question = example["question"]
        answer = example["answer"]

        # 提取最终答案（GSM8K的答案格式为：推理过程\n#### 最终答案）
        if "####" in answer:
            reasoning, final_answer = answer.split("####")
            reasoning = reasoning.strip()
            final_answer = final_answer.strip()
        else:
            reasoning = answer
            final_answer = ""

        # prompt 保持不变
        prompt = f"Question: {question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

        # 根据 is_think 参数构建不同的 completion
        if self.is_think:
            # 思维链格式：在 completion 开头添加结构化思考引导
            completion = f"""<think>\n{reasoning}\n<think>\n\nFinal Answer: \\boxed{final_answer}"""
        else:
            # 标准格式
            completion = f"\n{reasoning}\n\nFinal Answer: \\boxed{{{final_answer}}}"

        return {
            "prompt": prompt,
            "completion": completion,
            "text": prompt + completion,  # 用于某些trainer
        }

    def format_for_rl(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """格式化为RL训练格式(Standard Format with Chat Template Applied)

        Args:
            example: 原始数据样本

        Returns:
            格式化后的样本，使用standard format_type (已应用chat template)
            - prompt: 应用chat template后的文本字符串
            - ground_truth: 正确答案
            - question: 原始问题
            - full_answer: 完整答案
        """
        question = example["question"]
        answer = example["answer"]

        # 提取最终答案
        if "####" in answer:
            _, final_answer = answer.split("####")
            final_answer = final_answer.strip()
        else:
            final_answer = answer.strip()

        # prompt 保持不变（无论 is_think 参数如何）
        prompt_content = f"Question: {question}\n\nLet's solve this step by step:"

        # 如果提供了tokenizer,应用chat template
        if self.tokenizer:
            messages = [{"role": "user", "content": prompt_content}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # 如果没有tokenizer,直接使用原始文本
            prompt_text = prompt_content

        return {
            "prompt": prompt_text,  # Standard format_type (string)
            "ground_truth": final_answer,
            "question": question,
            "full_answer": answer,
        }
