from typing import Optional, Union

from ..core.config import Config
from ..core.client import Client
from ..tools.registry import ToolRegistry
from ..tools.builtin.planner import write_todos
from ..tools.builtin.think_tool import think
from .openai_func_call_agent import OpenAIFuncCallAgent

DEFAULT_PLAN_SOLVE_PROMPT = """你是一个规划求解 Agent。遵循以下模式：

## 工作流程

1. **思考** - 使用 think 深入分析任务，识别关键要素、评估方案、规划执行步骤
2. **规划** - 使用 write_todos 创建任务列表
3. **执行** - 逐步完成任务，更新状态
4. **完成** - 标记所有任务为完成并提供最终答案

## 工具使用说明

### think
在开始任务前，先调用此工具进行系统性思考：
- 拆解问题，识别关键要素
- 评估方案，规划执行步骤
- 发现新信息时，可再次调用更新思考

### write_todos
用于追踪任务进度：
- 任务开始时立即调用，将首个任务标记为 'in_progress'，其他任务标记为 'pending'
- 完成每个步骤后，更新状态为 'completed'，将下一个任务标记为 'in_progress'
- 保持至少有一个任务处于 'in_progress' 状态
- 根据进展动态调整任务列表

## 注意事项
- 工具不要并行调用
- 灵活调整任务列表，适应新信息"""


class PlanSolveAgent(OpenAIFuncCallAgent):
    """Plan-Solve Agent，继承 OpenAIFuncCallAgent，默认配置 write_todos 工具"""

    def __init__(
        self,
        name: str = "PlanSolveAgent",
        client: Client = Client(),
        system_prompt: str = DEFAULT_PLAN_SOLVE_PROMPT,
        config: Optional[Config] = None,
        tool_registry: ToolRegistry = ToolRegistry(),
        tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 20,
    ):
        super().__init__(
            name=name,
            client=client,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry,
            tool_choice=tool_choice,
            max_tool_iterations=max_tool_iterations,
        )
        self.add_tool(write_todos)
        self.add_tool(think)
