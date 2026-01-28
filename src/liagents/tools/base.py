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
        """执行工具

        Args:
            parameters: 参数字典

        Returns:
            执行结果字符串
        """
        # 转换为适合函数调用的参数
        kwargs = {}
        for param in self.get_parameters():
            if param.name in parameters:
                kwargs[param.name] = parameters[param.name]

        try:
            result = self.func(**kwargs)
            # 如果结果是字符串，直接返回
            if isinstance(result, str):
                return result
            # 否则转换为字符串
            return str(result)
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
        """转换为 OpenAI function calling schema 格式

        用于 FunctionCallAgent，使工具能够被 OpenAI 原生 function calling 使用

        Returns:
            符合 OpenAI function calling 标准的 schema
        """
        parameters = self.get_parameters()

        # 构建 properties
        properties = {}
        required = []

        for param in parameters:
            # 基础属性定义
            prop = {"type": param.type, "description": param.description}

            # 如果有默认值，添加到描述中（OpenAI schema 不支持 default 字段）
            if param.default is not None:
                prop["description"] = f"{param.description} (默认: {param.default})"

            # 如果是数组类型，添加 items 定义
            if param.type == "array":
                prop["items"] = {"type": "string"}  # 默认字符串数组

            properties[param.name] = prop

            # 收集必需参数
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

    自动使用函数名作为工具名，使用 docstring 作为工具描述。
    自动从函数签名推断参数类型和必需性。

    支持使用 PEP 593 的 Annotated 类型注解来提供参数描述。

    Args:
        func: 被装饰的函数

    Returns:
        Tool 实例

    Examples:
        >>> from typing import Annotated
        >>>
        >>> @tool
        >>> def search(
        >>>     query: Annotated[str, "搜索查询关键词"],
        >>>     limit: Annotated[int, "返回结果数量上限"] = 10
        >>> ) -> str:
        ...     \"\"\"搜索工具，支持指定结果数量\"\"\"
        ...     return f"搜索 {query}，返回 {limit} 条结果"

        >>> # 使用
        >>> search.run({"query": "Python"})
        >>> search.run({"query": "Python", "limit": 5})

    不使用 Annotated 的示例（向后兼容）：
        >>> @tool
        >>> def simple_search(query: str, limit: int = 10) -> str:
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
            # 默认为 string
            return "string"

    def _infer_parameters(func: Callable) -> List[ToolParameter]:
        """从函数签名推断参数定义

        支持使用 PEP 593 的 Annotated 类型注解来提供参数描述：

        Examples:
            >>> def search(
            ...     query: Annotated[str, "搜索查询关键词"],
            ...     limit: Annotated[int, "返回结果数量上限"] = 10
            ... ) -> str:
            ...     pass
        """
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            # 跳过 self 参数
            if param_name == "self":
                continue

            # 提取类型和描述
            param_type = "string"
            param_description = f"{param_name} 参数"

            if param.annotation != inspect.Parameter.empty:
                annotation = param.annotation

                # 检查是否是 Annotated 类型
                if get_origin(annotation) is Annotated:
                    args = get_args(annotation)
                    if len(args) >= 2:
                        # 第一个参数是实际类型
                        actual_type = args[0]
                        # 第二个参数是描述（通常是字符串）
                        description = args[1] if isinstance(args[1], str) else None

                        # 推断类型
                        param_type = _infer_type_string(actual_type)

                        # 使用 Annotated 中的描述
                        if description:
                            param_description = description
                else:
                    # 不是 Annotated，直接推断类型
                    param_type = _infer_type_string(annotation)

            # 判断是否必需
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
