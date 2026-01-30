"""RL训练核心功能模块

包含SFT和GRPO训练算法的具体实现。
"""

from typing import Optional
from datasets import Dataset

from .utils import check_trl_installation


def setup_training_environment(
    output_dir: str,
    seed: int = 42,
) -> None:
    """设置训练环境

    Args:
        output_dir: 输出目录
        seed: 随机种子
    """
    import os
    import random
    import numpy as np

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置随机种子
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    random.seed(seed)
    np.random.seed(seed)

    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"训练环境设置完成")
    print(f"   - 输出目录: {output_dir}")
    print(f"   - 随机种子: {seed}")
