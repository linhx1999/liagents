"""测试 Tool 基类和装饰器"""

import pytest
from unittest.mock import Mock
from typing import Annotated

from liagents.tools.base import Tool, ToolParameter, tool


class TestToolParameter:
    """测试 ToolParameter 数据类"""

    def test_init_basic(self):
        """测试基本初始化"""
        param = ToolParameter(
            name="test_param",
            type="string",
            description="测试参数",
        )
        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == "测试参数"
        assert param.required is True
        assert param.default is None
        assert param.items_type == "string"

    def test_init_with_default(self):
        """测试带默认值的初始化"""
        param = ToolParameter(
            name="test_param",
            type="integer",
            description="测试参数",
            required=False,
            default=10,
        )
        assert param.required is False
        assert param.default == 10

    def test_init_with_items_type(self):
        """测试带数组元素类型的初始化"""
        param = ToolParameter(
            name="tags",
            type="array",
            description="标签列表",
            items_type="integer",
        )
        assert param.type == "array"
        assert param.items_type == "integer"


class TestToolInit:
    """测试 Tool 初始化"""

    def test_init_basic(self):
        """测试基本初始化"""

        def dummy_func():
            pass

        param = ToolParameter(name="param1", type="string", description="参数1")
        tool = Tool(
            func=dummy_func,
            name="test_tool",
            description="测试工具",
            parameters=[param],
        )

        assert tool.name == "test_tool"
        assert tool.description == "测试工具"
        assert tool.func == dummy_func
        assert len(tool.get_parameters()) == 1

    def test_init_multiple_params(self):
        """测试多参数初始化"""

        def dummy_func(a, b, c):
            pass

        params = [
            ToolParameter(name="a", type="string", description="参数A"),
            ToolParameter(
                name="b", type="integer", description="参数B", required=False, default=0
            ),
            ToolParameter(name="c", type="boolean", description="参数C"),
        ]
        test_tool = Tool(
            func=dummy_func, name="multi_tool", description="多参数工具", parameters=params
        )

        assert len(test_tool.get_parameters()) == 3


class TestToolGetParameters:
    """测试 Tool 参数获取"""

    def test_get_parameters(self):
        """测试获取参数"""

        def dummy_func(x, y):
            pass

        params = [
            ToolParameter(name="x", type="string", description="X参数"),
            ToolParameter(name="y", type="integer", description="Y参数"),
        ]
        test_tool = Tool(
            func=dummy_func, name="test", description="测试", parameters=params
        )

        result = test_tool.get_parameters()
        assert len(result) == 2
        assert result[0].name == "x"
        assert result[1].name == "y"

    def test_get_parameters_empty(self):
        """测试无参数时获取"""

        def dummy_func():
            pass

        test_tool = Tool(func=dummy_func, name="test", description="测试", parameters=[])
        result = test_tool.get_parameters()
        assert result == []


class TestToolRun:
    """测试 Tool 执行"""

    def test_run_success(self):
        """测试成功执行"""

        def add(a, b):
            return a + b

        params = [
            ToolParameter(name="a", type="integer", description="A"),
            ToolParameter(name="b", type="integer", description="B"),
        ]
        test_tool = Tool(func=add, name="add", description="加法工具", parameters=params)

        result = test_tool.run({"a": 1, "b": 2})
        assert result == "3"

    def test_run_with_missing_params(self):
        """测试缺少参数时执行"""

        def add(a, b):
            return a + b

        params = [
            ToolParameter(name="a", type="integer", description="A"),
            ToolParameter(name="b", type="integer", description="B"),
        ]
        test_tool = Tool(func=add, name="add", description="加法工具", parameters=params)

        # 缺少参数时，kwargs 中没有对应的 key，func 会使用默认参数
        # 如果没有默认参数会报错
        result = test_tool.run({})
        assert "执行出错" in result or "TypeError" in result

    def test_run_with_extra_params(self):
        """测试带额外参数时执行"""

        def add(a, b):
            return a + b

        params = [
            ToolParameter(name="a", type="integer", description="A"),
            ToolParameter(name="b", type="integer", description="B"),
        ]
        test_tool = Tool(func=add, name="add", description="加法工具", parameters=params)

        result = test_tool.run({"a": 1, "b": 2, "c": 3})
        assert result == "3"

    def test_run_returns_string(self):
        """测试执行结果转换为字符串"""

        def get_value():
            return 42

        test_tool = Tool(
            func=get_value, name="get_value", description="获取值", parameters=[]
        )

        result = test_tool.run({})
        assert result == "42"

    def test_run_exception(self):
        """测试执行异常"""

        def raise_error():
            raise ValueError("测试错误")

        test_tool = Tool(
            func=raise_error, name="raise_error", description="报错工具", parameters=[]
        )

        result = test_tool.run({})
        assert "执行出错" in result
        assert "测试错误" in result


class TestToolValidateParameters:
    """测试 Tool 参数验证"""

    def test_validate_valid_params(self):
        """测试有效参数验证"""

        def func(a, b):
            pass

        params = [
            ToolParameter(name="a", type="string", description="A", required=True),
            ToolParameter(name="b", type="integer", description="B", required=False),
        ]
        test_tool = Tool(func=func, name="test", description="测试", parameters=params)

        assert test_tool.validate_parameters({"a": "value"}) is True
        assert test_tool.validate_parameters({"a": "value", "b": 1}) is True

    def test_validate_missing_required(self):
        """测试缺少必需参数"""

        def func(a, b):
            pass

        params = [
            ToolParameter(name="a", type="string", description="A", required=True),
            ToolParameter(name="b", type="integer", description="B", required=True),
        ]
        test_tool = Tool(func=func, name="test", description="测试", parameters=params)

        assert test_tool.validate_parameters({"a": "value"}) is False
        assert test_tool.validate_parameters({"b": 1}) is False

    def test_validate_empty_params(self):
        """测试空参数验证"""

        def func():
            pass

        test_tool = Tool(func=func, name="test", description="测试", parameters=[])

        assert test_tool.validate_parameters({}) is True


class TestToolToDict:
    """测试 Tool 转换为字典"""

    def test_to_dict_basic(self):
        """测试基本转换"""

        def func(a):
            pass

        params = [
            ToolParameter(name="a", type="string", description="A参数", required=True)
        ]
        test_tool = Tool(
            func=func, name="test_tool", description="测试工具", parameters=params
        )

        result = test_tool.to_dict()

        assert result["name"] == "test_tool"
        assert result["description"] == "测试工具"
        assert "parameters" in result
        assert len(result["parameters"]) == 1
        assert result["parameters"][0]["name"] == "a"

    def test_to_dict_all_fields(self):
        """测试所有字段的转换"""

        def func(a, b, c):
            pass

        params = [
            ToolParameter(name="a", type="string", description="A", required=True),
            ToolParameter(
                name="b", type="integer", description="B", required=False, default=10
            ),
            ToolParameter(name="c", type="array", description="C", items_type="string"),
        ]
        test_tool = Tool(func=func, name="test", description="测试", parameters=params)

        result = test_tool.to_dict()
        params_result = result["parameters"]

        assert params_result[0]["required"] is True
        assert params_result[1]["default"] == 10
        assert params_result[2]["items_type"] == "string"


class TestToolToSchema:
    """测试 Tool 转换为 OpenAI Schema"""

    def test_to_schema_basic(self):
        """测试基本转换"""

        def func(query):
            pass

        params = [
            ToolParameter(
                name="query", type="string", description="搜索查询", required=True
            )
        ]
        test_tool = Tool(
            func=func, name="search", description="搜索工具", parameters=params
        )

        result = test_tool.to_schema()

        assert result["type"] == "function"
        assert "function" in result
        assert result["function"]["name"] == "search"
        assert result["function"]["description"] == "搜索工具"

    def test_to_schema_parameters(self):
        """测试参数转换"""

        def func(a, b):
            pass

        params = [
            ToolParameter(name="a", type="string", description="字符串参数", required=True),
            ToolParameter(
                name="b", type="integer", description="整数参数", required=False, default=0
            ),
        ]
        test_tool = Tool(func=func, name="test", description="测试", parameters=params)

        result = test_tool.to_schema()
        schema_params = result["function"]["parameters"]

        assert schema_params["type"] == "object"
        assert "properties" in schema_params
        assert "a" in schema_params["properties"]
        assert "b" in schema_params["properties"]
        assert "required" in schema_params
        assert "a" in schema_params["required"]

    def test_to_schema_array_type(self):
        """测试数组类型转换"""

        def func(items):
            pass

        params = [
            ToolParameter(
                name="items", type="array", description="列表", items_type="string"
            )
        ]
        test_tool = Tool(func=func, name="test", description="测试", parameters=params)

        result = test_tool.to_schema()
        items_prop = result["function"]["parameters"]["properties"]["items"]

        assert items_prop["type"] == "array"
        assert items_prop["items"]["type"] == "string"


class TestToolStr:
    """测试 Tool 字符串表示"""

    def test_str_representation(self):
        """测试字符串表示"""

        def func():
            pass

        test_tool = Tool(func=func, name="test_tool", description="测试", parameters=[])

        result = str(test_tool)
        assert result == "Tool(name=test_tool)"

    def test_repr(self):
        """测试 repr"""

        def func():
            pass

        test_tool = Tool(func=func, name="test_tool", description="测试", parameters=[])

        result = repr(test_tool)
        assert result == "Tool(name=test_tool)"


class TestToolDecorator:
    """测试 @tool 装饰器"""

    def test_decorator_basic(self):
        """测试基本装饰器使用"""

        @tool
        def search(query: str) -> str:
            """搜索工具"""
            return f"搜索: {query}"

        assert isinstance(search, Tool)
        assert search.name == "search"
        assert search.description == "搜索工具"

    def test_decorator_with_annotated_params(self):
        """测试带 Annotated 参数的装饰器"""

        @tool
        def calculate(
            expression: Annotated[str, "计算表达式"], precision: Annotated[int, "精度"] = 2
        ) -> str:
            """计算工具"""
            return expression

        assert isinstance(calculate, Tool)
        params = calculate.get_parameters()
        assert len(params) == 2

        # 检查第一个参数
        assert params[0].name == "expression"
        assert params[0].type == "string"
        assert params[0].description == "计算表达式"
        assert params[0].required is True

        # 检查第二个参数
        assert params[1].name == "precision"
        assert params[1].type == "integer"
        assert params[1].description == "精度"
        assert params[1].required is False
        assert params[1].default == 2

    def test_decorator_type_inference(self):
        """测试类型推断"""

        @tool
        def types_demo(
            s: str,
            i: int,
            f: float,
            b: bool,
            lst: list,
        ) -> str:
            """类型演示"""
            return "ok"

        tool_instance = types_demo
        params = tool_instance.get_parameters()

        assert params[0].type == "string"
        assert params[1].type == "integer"
        assert params[2].type == "number"
        assert params[3].type == "boolean"
        assert params[4].type == "array"

    def test_decorator_no_docstring(self):
        """测试没有 docstring 的情况"""

        @tool
        def no_doc():
            return "result"

        assert no_doc.description == "no_doc 工具"

    def test_decorator_execute(self):
        """测试装饰器创建的 tool 执行"""

        @tool
        def add(a: int, b: int) -> int:
            """加法工具"""
            return a + b

        result = add.run({"a": 5, "b": 3})
        assert result == "8"

    def test_decorator_with_list_generic(self):
        """测试带 List 泛型的参数"""

        @tool
        def process_items(items: list[str]) -> str:
            """处理列表"""
            return str(items)

        params = process_items.get_parameters()
        assert params[0].type == "array"
        assert params[0].items_type == "string"

    def test_decorator_with_dict(self):
        """测试 dict 类型参数"""

        @tool
        def update_config(config: dict) -> str:
            """更新配置"""
            return str(config)

        params = update_config.get_parameters()
        assert params[0].type == "object"
