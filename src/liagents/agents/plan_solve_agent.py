from typing import Optional, Union, Any

from ..core.config import Config
from ..core.client import Client
from ..tools.registry import ToolRegistry
from ..tools.builtin.write_todos import write_todos
from .openai_func_call_agent import OpenAIFuncCallAgent

DEFAULT_PLAN_SOLVE_PROMPT = """You are a Plan-Solve Agent. Follow this pattern:

1. PLAN: When given a task, first create a todo list using write_todos tool
2. EXECUTE: Work through each item, update status as you progress
3. COMPLETE: Mark items as completed and provide final answer

Use write_todos to track progress. Format todo items with status field.

## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant."""


class PlanSolveAgent(OpenAIFuncCallAgent):
    """Plan-Solve Agent，继承 OpenAIFuncCallAgent，默认配置 write_todos 工具"""

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
        self.add_tool(write_todos)
