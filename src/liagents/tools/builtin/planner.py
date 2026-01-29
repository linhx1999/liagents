from typing import Annotated

from ..base import tool


@tool
def write_todos(
    todo_list: Annotated[
        list[dict[str, str]],
        "待办事项列表，每个元素是一个字典，包含 'content' 和 'status' 字段",
    ]
) -> str:
    """用于规划和追踪复杂任务的进度。开始任何任务前，必须首先调用此工具创建待办列表。在执行任务过程中，持续更新待办列表状态。"""
    return todo_list
