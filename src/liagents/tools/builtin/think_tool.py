from typing import Annotated

from ..base import tool


@tool
def think(
    thinking: Annotated[
        str,
        "当前推理/思考内容，可以是问题分析、方案评估、步骤规划等任何需要思考的内容",
    ]
) -> str:
    """用于推理和分析的工具。在任务开始、规划行动前或使用别的工具后，都应该调用此工具进行系统性思考。"""
    return thinking
