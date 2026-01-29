from typing import Optional, Union, Any

from ..core.config import Config
from ..core.client import Client
from ..tools.registry import ToolRegistry
from ..tools.builtin.write_todo import write_todo
from .openai_func_call_agent import OpenAIFuncCallAgent

DEFAULT_PLAN_SOLVE_PROMPT = """You are a Plan-Solve Agent. Follow this pattern:

1. PLAN: When given a task, first create a todo list using write_todo tool
2. EXECUTE: Work through each item, update status as you progress
3. COMPLETE: Mark items as completed and provide final answer

Use write_todo to track progress. Format todo items with status field."""


class PlanSolveAgent(OpenAIFuncCallAgent):
    """Plan-Solve Agent，继承 OpenAIFuncCallAgent，默认配置 write_todo 工具"""

    def __init__(
        self,
        name: str = "PlanSolveAgent",
        client: Client = Client(),
        system_prompt: str = DEFAULT_PLAN_SOLVE_PROMPT,
        config: Optional[Config] = None,
        tool_registry: ToolRegistry = ToolRegistry(),
        default_tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 20,
    ):
        super().__init__(
            name=name,
            client=client,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry,
            default_tool_choice=default_tool_choice,
            max_tool_iterations=max_tool_iterations,
        )
        self.add_tool(write_todo)
