from typing import Annotated

from ..base import tool


@tool
def write_todos(
    todo_list: Annotated[
        list[dict[str, str]],
        "待办事项列表，例: [{'content': '任务1', 'status': 'completed'}, {'content': '任务2', 'status': 'in_progress'}, {'content': '任务3', 'status': 'pending'}]",
    ]
) -> str:
    """
    规划和追踪任务进度的工具。

    使用时机：
    - 收到新任务时，调用创建待办列表
    - 执行任务过程中，根据任务进度，持续更新状态
    - 完成一个任务后，更新其状态为 completed
    - 开始执行某任务时，更新其状态为 in_progress
    """
    return str(todo_list)
