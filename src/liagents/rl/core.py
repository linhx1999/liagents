"""RL训练核心功能模块

包含SFT和GRPO训练算法的具体实现。
"""

from typing import Optional
from datasets import Dataset

from .utils import check_trl_installation
