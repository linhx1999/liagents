from typing import Annotated

from ..base import tool


@tool
def think_tool(
    thought: Annotated[
        str,
        "当前推理/思考内容，可以是问题分析、方案评估、步骤规划等任何需要思考的内容",
    ]
) -> str:
    """用于深度推理和分析问题的工具。在面对复杂问题、做出决策或规划行动前,应该先调用此工具进行系统性思考。"""
    return thought
