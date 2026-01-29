from typing import Optional, Union

from ..core.config import Config
from ..core.client import Client
from ..tools.registry import ToolRegistry
from ..tools.builtin import think
from .openai_func_call_agent import OpenAIFuncCallAgent


class ReflectionAgent(OpenAIFuncCallAgent):
    def __init__(
        self,
        name: str = "ReflectionAgent",
        client: Client = Client(),
        system_prompt: str = "",
        config: Optional[Config] = None,
        tool_registry: ToolRegistry = ToolRegistry(),
        tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 20,
        debug: bool = False,
    ):
        super().__init__(
            name=name,
            client=client,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry,
            tool_choice=tool_choice,
            max_tool_iterations=max_tool_iterations,
            debug=debug,
        )
        self.add_tool(think)
