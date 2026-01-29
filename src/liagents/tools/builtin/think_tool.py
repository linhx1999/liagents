from typing import Annotated

from ..base import tool


@tool
def think(
    thinking: Annotated[
        str,
        "推理内容，如：问题分析、方案评估、步骤规划、自我反思等",
    ]
) -> str:
    """
    用于推理和分析的工具。

    使用时机：
    - 收到新任务时，先思考进行任务规划，确认待办事项
    - 遇到困难时，反思问题并调整策略
    - 使用其他工具后，思考结果并决定下一步
    - 在完成任务的最后一步，思考确认任务是否完成
    """
    return thinking
