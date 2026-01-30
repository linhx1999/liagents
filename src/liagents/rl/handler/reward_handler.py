"""RL奖励函数处理模块

包含奖励函数创建和管理的相关功能。
"""

from typing import Dict, Any
import json


class RLRewardHandler:
    """RL奖励处理类，负责奖励函数的创建和管理"""

    def __init__(self):
        # 存储自定义奖励函数
        self.custom_reward_functions = {}

    def register_reward_function(self, name: str, reward_fn) -> None:
        """
        注册自定义奖励函数

        Args:
            name: 奖励函数名称
            reward_fn: 奖励函数(接受completions和kwargs,返回rewards列表)
        """
        self.custom_reward_functions[name] = reward_fn
        print(f"✅ 已注册自定义奖励函数: {name}")

    def handle_create_reward(self, parameters: Dict[str, Any]) -> str:
        """处理奖励函数创建操作"""
        from hello_agents.rl import (
            create_accuracy_reward,
            create_length_penalty_reward,
            create_step_reward,
        )

        reward_type = parameters.get("reward_type", "accuracy").lower()

        if reward_type == "accuracy":
            reward_fn = create_accuracy_reward()
            result = {
                "status": "success",
                "reward_type": "accuracy",
                "description": "准确性奖励函数: 答案正确=1.0, 错误=0.0",
            }
        elif reward_type == "length_penalty":
            penalty_weight = parameters.get("penalty_weight", 0.001)
            max_length = parameters.get("max_length", 1024)
            # 创建基础奖励函数
            base_reward_fn = create_accuracy_reward()
            reward_fn = create_length_penalty_reward(
                base_reward_fn=base_reward_fn,
                max_length=max_length,
                penalty_weight=penalty_weight,
            )
            result = {
                "status": "success",
                "reward_type": "length_penalty",
                "penalty_weight": penalty_weight,
                "max_length": max_length,
                "description": f"长度惩罚奖励函数: 基础奖励 - {penalty_weight} * (长度 / {max_length})",
            }
        elif reward_type == "step":
            step_bonus = parameters.get("step_bonus", 0.1)
            # 创建基础奖励函数
            base_reward_fn = create_accuracy_reward()
            reward_fn = create_step_reward(
                base_reward_fn=base_reward_fn, step_bonus=step_bonus
            )
            result = {
                "status": "success",
                "reward_type": "step",
                "step_bonus": step_bonus,
                "description": f"步骤奖励函数: 基础奖励 + {step_bonus} * 步骤数",
            }
        else:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"不支持的奖励类型: {reward_type}。支持的类型: accuracy, length_penalty, step",
                },
                ensure_ascii=False,
                indent=2,
            )

        return json.dumps(result, ensure_ascii=False, indent=2)
