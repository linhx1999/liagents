"""RL数据集处理模块

包含数据集加载和处理的相关功能。
"""

from typing import Dict, Any
import json


class RLDataHandler:
    """RL数据处理类，负责数据集的加载和管理"""
    
    def __init__(self):
        # 存储自定义数据集
        self.custom_datasets = {}

    def register_dataset(self, name: str, dataset) -> None:
        """
        注册自定义数据集

        Args:
            name: 数据集名称
            dataset: 数据集对象(HuggingFace Dataset)
        """
        self.custom_datasets[name] = dataset
        print(f"✅ 已注册自定义数据集: {name}")

    def handle_load_dataset(self, parameters: Dict[str, Any]) -> str:
        """处理数据集加载操作"""
        from hello_agents.rl import create_sft_dataset, create_rl_dataset

        format_type = parameters.get("format", "sft").lower()
        split = parameters.get("split", "train")
        max_samples = parameters.get("max_samples", 100)
        model_name = parameters.get("model_name", "Qwen/Qwen3-0.6B")

        if format_type == "sft":
            dataset = create_sft_dataset(split=split, max_samples=max_samples)
        elif format_type == "rl":
            dataset = create_rl_dataset(split=split, max_samples=max_samples, model_name=model_name)
        else:
            return json.dumps({
                "status": "error",
                "message": f"不支持的数据格式: {format_type}。支持的格式: sft, rl"
            }, ensure_ascii=False, indent=2)

        result = {
            "status": "success",
            "format": format_type,
            "split": split,
            "dataset_size": len(dataset),
            "sample_keys": list(dataset[0].keys()) if len(dataset) > 0 else []
        }
        return json.dumps(result, ensure_ascii=False, indent=2)