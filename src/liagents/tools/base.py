"""工具基类和装饰器定义"""

import inspect
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Dict, List, get_args, get_origin


@dataclass
class ToolParameter:
    """工具参数定义"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool:
    """工具类，用于包装普通函数为工具"""

    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: List[ToolParameter],
    ):
        """初始化工具

        Args:
            func: 被包装的函数
            name: 工具名称
            description: 工具描述
            parameters: 参数定义列表（由装饰器自动推断）
        """
        self.func = func
        self.name = name
        self.description = description
        self._parameters = parameters

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        return self._parameters

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具并返回结果字符串"""
        kwargs = {p.name: parameters[p.name] for p in self.get_parameters() if p.name in parameters}
        try:
            result = self.func(**kwargs)
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            return f"执行出错: {str(e)}"

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        required_params = [p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in self.get_parameters()
            ],
        }

    def to_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling schema 格式"""
        properties = {}
        required = []

        for param in self.get_parameters():
            prop = {"type": param.type, "description": param.description}
            if param.default is not None:
                prop["description"] = f"{param.description} (默认: {param.default})"
            if param.type == "array":
                prop["items"] = {"type": "string"}
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()


def tool(func: Callable) -> Tool:
    """极简工具装饰器，将函数转换为工具

    自动使用函数名作为工具名、docstring 作为工具描述，
    并从函数签名推断参数类型和必需性。

    支持使用 PEP 593 的 Annotated 类型注解提供参数描述。

    Examples:
        >>> from typing import Annotated
        >>> @tool
        >>> def search(
        ...     query: Annotated[str, "搜索查询关键词"],
        ...     limit: Annotated[int, "返回结果数量上限"] = 10
        ... ) -> str:
        ...     \"\"\"搜索工具\"\"\"
        ...     return f"搜索 {query}，返回 {limit} 条结果"
    """
    def _infer_type_string(type_annotation: Any) -> str:
        """将 Python 类型注解转换为 JSON Schema 类型字符串"""
        if type_annotation == int:
            return "integer"
        elif type_annotation == float:
            return "number"
        elif type_annotation == bool:
            return "boolean"
        elif type_annotation == str:
            return "string"
        elif type_annotation == list:
            return "array"
        elif get_origin(type_annotation) is list:
            return "array"
        else:
            return "string"

    def _infer_parameters(func: Callable) -> List[ToolParameter]:
        """从函数签名推断参数定义，支持 Annotated 类型注解"""
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = "string"
            param_description = f"{param_name} 参数"

            if param.annotation != inspect.Parameter.empty:
                annotation = param.annotation
                if get_origin(annotation) is Annotated:
                    args = get_args(annotation)
                    if len(args) >= 2:
                        actual_type = args[0]
                        description = args[1] if isinstance(args[1], str) else None
                        param_type = _infer_type_string(actual_type)
                        if description:
                            param_description = description
                else:
                    param_type = _infer_type_string(annotation)

            required = param.default == inspect.Parameter.empty

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_description,
                    required=required,
                    default=param.default if not required else None,
                )
            )

        return parameters

    tool_name = func.__name__
    tool_description = (func.__doc__ or f"{tool_name} 工具").strip()
    tool_parameters = _infer_parameters(func)

    return Tool(
        func=func,
        name=tool_name,
        description=tool_description,
        parameters=tool_parameters,
    )
