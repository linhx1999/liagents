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
    - 在执行任务前，必须先进行任务规划，调用该工具创建待办事项列表（初始化全部任务为 pending 状态）
    - 执行任务过程中，根据任务进度，持续更新状态
    - 完成一个任务后，更新其状态为 completed
    - 开始执行某任务时，更新其状态为 in_progress

    注意：
    - 若发现代办事项有误（如任务描述错误、任务时间冲突等），可以且必须及时更新其内容和状态
    - 调用该工具时，返回更新后的待办事项列表
    """
    return str(todo_list)
