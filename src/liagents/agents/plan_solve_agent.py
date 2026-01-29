from typing import Optional, Union

from ..core.config import Config
from ..core.client import Client
from ..tools.registry import ToolRegistry
from ..tools.builtin.planner import write_todos
from ..tools.builtin.think_tool import think
from .openai_func_call_agent import OpenAIFuncCallAgent


class PlanSolveAgent(OpenAIFuncCallAgent):
    def __init__(
        self,
        name: str = "PlanSolveAgent",
        client: Client = Client(),
        system_prompt: str = "",
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
