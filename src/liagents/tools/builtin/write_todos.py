"""待办事项工具 - 支持创建和更新待办列表"""

from typing import Annotated

from ..base import tool


@tool
def write_todos(
    todo_list: Annotated[
        list[dict[str, str]],
        "待办事项列表，每项格式: {'content': '任务内容', 'status': 'pending|in_progress|completed'}",
    ]
) -> str:
    """用于规划和追踪复杂任务的进度。开始任何任务前，必须首先调用此工具创建待办列表。在执行任务过程中，持续更新待办列表状态。

每个待办事项包含两个必需字段:
- content: 任务描述（字符串）
- status: 任务状态，必须是以下之一:
  * 'pending' - 待完成
  * 'in_progress' - 进行中
  * 'completed' - 已完成

## 使用规则
**任务开始前：** 立即调用此工具，将所有步骤列为待办项，并将首个任务标记为 'in_progress'

**任务执行中：**
1. 完成每个步骤后，立即更新状态为 'completed'，并标记下一个任务为 'in_progress'
2. 保持至少有一个任务处于 'in_progress' 状态（除非全部完成）
3. 发现新步骤时，添加新待办项；发现无关步骤时，移除相应待办项
4. 遇到阻塞时，将当前任务保持为 'in_progress' 并创建新任务描述如何解决

## 调用示例
```json
[{"content": "分析问题并制定计划", "status": "in_progress"}, {"content": "逐步解决问题", "status": "pending"}, {"content": "验证答案正确性", "status": "pending"}]
```
"""
    return str(todo_list)
