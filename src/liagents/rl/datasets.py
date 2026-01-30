from typing import Dict, Any, List, Optional, Literal
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from trl import apply_chat_template


def create_dataset(
    dataset_name_or_path: str = "openai/gsm8k",
    format_type: Literal["sft", "rl"] = "sft",
    max_samples: Optional[int] = 100,
    split: str = "train",
    tokenizer: Optional[AutoTokenizer] = None
) -> Dataset:
    """创建数据集

    Args:
        dataset_name_or_path: 数据集名称或路径
        format_type: 数据格式 (sft/rl)
        max_samples: 最大样本数
        split: 数据集分割
        tokenizer: Tokenizer对象

    Returns:
        格式化后的数据集
    """
    if "gsm8k" in dataset_name_or_path.lower().strip():
        dataset_wrapper = GSM8KDataset(
            dataset_name_or_path=dataset_name_or_path,
            split=split,
            max_samples=max_samples,
            format_type=format_type,
            tokenizer=tokenizer
        )
        return dataset_wrapper.get_dataset()
    else:
        raise ValueError(f"不支持的数据集: {dataset_name_or_path}")


class GSM8KDataset:
    """GSM8K数学推理数据集

    GSM8K (Grade School Math 8K) 是一个包含8500个高质量小学数学问题的数据集。
    每个问题都需要2-8步的推理过程来解决。
    """

    def __init__(
        self,
        dataset_name_or_path: str = "openai/gsm8k",
        split: str = "train",
        max_samples: int = 100,
        format_type: Literal["sft", "rl"] = "sft",
        tokenizer: Optional[AutoTokenizer] = None  # 用于RL格式应用chat template
    ):
        """
        初始化GSM8K数据集

        Args:
            dataset_name_or_path: 数据集名称或路径
            split: 数据集分割 ("train" 或 "test")
            max_samples: 最大样本数（用于快速测试）
            format_type: 数据格式类型 ("sft" 用于监督学习, "rl" 用于强化学习)
            tokenizer: Tokenizer对象,用于RL格式应用chat template
        """
        self.split = split
        self.max_samples = max_samples
        self.format_type = format_type
        self.tokenizer = tokenizer

        print(f"加载 GSM8K 数据集 (split={split})...")
        self.dataset = load_dataset(dataset_name_or_path, "main", split=split)

        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            print(f"   使用 {len(self.dataset)} 个样本（限制：{max_samples}）")
        else:
            print(f"   加载了 {len(self.dataset)} 个样本")
    
    def format_for_sft(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        格式化为SFT训练格式
        
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
        
        # 构造prompt和completion
        prompt = f"Question: {question}\n\nLet's solve this step by step:\n"
        completion = f"{reasoning}\n\nFinal Answer: {final_answer}"
        
        return {
            "prompt": prompt,
            "completion": completion,
            "text": prompt + completion  # 用于某些trainer
        }
    
    def format_for_rl(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化为RL训练格式(Standard Format with Chat Template Applied)

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

        # 构造prompt内容
        prompt_content = f"Question: {question}\n\nLet's solve this step by step:"

        # 如果提供了tokenizer,应用chat template
        if self.tokenizer:
            messages = [{"role": "user", "content": prompt_content}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 如果没有tokenizer,直接使用原始文本
            prompt_text = prompt_content

        return {
            "prompt": prompt_text,  # Standard format_type (string)
            "ground_truth": final_answer,
            "question": question,
            "full_answer": answer
        }
    
    def get_dataset(self) -> Dataset:
        """
        获取格式化后的数据集

        Returns:
            HuggingFace Dataset对象
        """
        if self.format_type == "sft":
            formatted_dataset = self.dataset.map(
                self.format_for_sft,
                remove_columns=self.dataset.column_names
            )
        elif self.format_type == "rl":
            formatted_dataset = self.dataset.map(
                self.format_for_rl,
                remove_columns=self.dataset.column_names
            )
        else:
            raise ValueError(f"不支持的格式类型: {self.format_type}")

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