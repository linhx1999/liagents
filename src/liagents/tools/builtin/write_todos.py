"""待办事项工具 - 支持创建和更新待办列表"""

from typing import Annotated

from ..base import tool


@tool
def write_todos(
    todo_list: Annotated[
        list[dict[str, str]],
        "待办事项列表，例如：[{'content': '任务1', 'status': 'completed'}, {'content': '任务2', 'status': 'in_progress'}, {'content': '任务3', 'status': 'pending'}]",
    ] = []
) -> str:
    """待办事项工具。创建或更新待办事项列表。

    每个事项包含 content（内容）和 status（状态）字段。
    status 可选值：pending（待完成）、in_progress（进行中）、completed（已完成）
    """
    return str(todo_list)
