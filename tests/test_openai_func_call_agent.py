"""测试 OpenAIFuncCallAgent"""

import pytest
import os
from unittest.mock import Mock
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

# 设置必需的环境变量
os.environ["MODEL"] = "test-model"
os.environ["OPENAI_API_KEY"] = "test-api-key"
os.environ["OPENAI_BASE_URL"] = "https://test.example.com"

from liagents.agents.openai_func_call_agent import OpenAIFuncCallAgent
from liagents.core.client import Client
from liagents.core.config import Config
from liagents.core.message import Message
from liagents.tools.registry import ToolRegistry
from liagents.tools.base import Tool, ToolParameter


# ========== 测试 Fixtures ==========


@pytest.fixture
def mock_client():
    """创建 Mock Client 实例"""
    client = Mock(spec=Client)
    return client


@pytest.fixture
def mock_config():
    """创建 Mock Config 实例"""
    return Config()


@pytest.fixture
def mock_tool():
    """创建 Mock Tool 实例"""
    tool = Mock(spec=Tool)
    tool.name = "test_tool"
    tool.description = "测试工具"
    tool.run = Mock(return_value="工具执行成功")
    tool.get_parameters = Mock(
        return_value=[
            ToolParameter(
                name="param1", type="string", description="参数1", required=True
            ),
            ToolParameter(
                name="param2", type="integer", description="参数2", required=False
            ),
        ]
    )
    # 让 to_schema 返回真实的 schema 字典
    tool.to_schema = Mock(
        return_value={
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "测试工具",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "参数1"},
                        "param2": {"type": "integer", "description": "参数2"},
                    },
                    "required": ["param1"],
                },
            },
        }
    )
    return tool


@pytest.fixture
def empty_tool_registry():
    """创建空的 ToolRegistry 实例"""
    return ToolRegistry()


@pytest.fixture
def mock_tool_registry(mock_tool):
    """创建包含工具的 ToolRegistry 实例"""
    registry = ToolRegistry()
    registry.register_tool(mock_tool)
    return registry


@pytest.fixture
def func_call_agent(mock_client, mock_config, empty_tool_registry):
    """创建 OpenAIFuncCallAgent 实例（不带工具）"""
    return OpenAIFuncCallAgent(
        name="test_agent",
        client=mock_client,
        system_prompt="你是一个测试助手",
        config=mock_config,
        tool_registry=empty_tool_registry,
    )


@pytest.fixture
def func_call_agent_with_tools(mock_client, mock_config, mock_tool_registry):
    """创建 OpenAIFuncCallAgent 实例（带工具）"""
    return OpenAIFuncCallAgent(
        name="test_agent_with_tools",
        client=mock_client,
        system_prompt="你是一个测试助手",
        config=mock_config,
        tool_registry=mock_tool_registry,
    )


@pytest.fixture
def mock_chat_completion_without_tools():
    """创建不带工具调用的模拟响应"""
    message = ChatCompletionMessage(
        role="assistant",
        content="你好！有什么可以帮助你的吗？",
    )
    choice = Choice(index=0, message=message, finish_reason="stop")
    completion = ChatCompletion(
        id="test-id",
        created=1234567890,
        model="test-model",
        choices=[choice],
        object="chat.completion",
    )
    return completion


@pytest.fixture
def mock_chat_completion_with_tool_call():
    """创建带工具调用的模拟响应"""
    tool_call = ChatCompletionMessageToolCall(
        id="call_123",
        type="function",
        function=Function(
            name="test_tool",
            arguments='{"param1": "value1"}',
        ),
    )
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
    )
    choice = Choice(index=0, message=message, finish_reason="tool_calls")
    completion = ChatCompletion(
        id="test-id",
        created=1234567890,
        model="test-model",
        choices=[choice],
        object="chat.completion",
    )
    return completion


# ========== 初始化测试 ==========


class TestOpenAIFuncCallAgentInit:
    """测试 OpenAIFuncCallAgent 初始化"""

    def test_init_basic(self, mock_client):
        """测试基本初始化"""
        agent = OpenAIFuncCallAgent(
            name="test_agent",
            client=mock_client,
            system_prompt="测试提示词",
        )

        assert agent.name == "test_agent"
        assert agent.client == mock_client
        assert agent.system_prompt == "测试提示词"
        assert agent.tool_registry is not None
        assert agent._history == []
        assert agent.tool_choice == "auto"
        assert agent.max_tool_iterations == 10

    def test_init_without_system_prompt(self, mock_client):
        """测试不提供系统提示词"""
        agent = OpenAIFuncCallAgent(name="test_agent", client=mock_client)

        assert agent.system_prompt == ""

    def test_init_with_tool_registry(self, mock_client, mock_tool_registry):
        """测试带工具注册表的初始化"""
        agent = OpenAIFuncCallAgent(
            name="test_agent",
            client=mock_client,
            tool_registry=mock_tool_registry,
        )

        assert agent.tool_registry == mock_tool_registry

    def test_init_with_custom_tool_choice(self, mock_client):
        """测试自定义 tool_choice"""
        agent = OpenAIFuncCallAgent(
            name="test_agent",
            client=mock_client,
            tool_choice="none",
        )

        assert agent.tool_choice == "none"

    def test_init_with_custom_max_iterations(self, mock_client):
        """测试自定义 max_tool_iterations"""
        agent = OpenAIFuncCallAgent(
            name="test_agent",
            client=mock_client,
            max_tool_iterations=5,
        )

        assert agent.max_tool_iterations == 5


# ========== _build_messages 测试 ==========


class TestBuildMessages:
    """测试 _build_messages 方法"""

    def test_build_messages_without_history(self, func_call_agent):
        """测试不带历史记录的消息构建"""
        messages = func_call_agent._build_messages("你好")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "你是一个测试助手"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "你好"

    def test_build_messages_with_history(self, func_call_agent):
        """测试带历史记录的消息构建"""
        func_call_agent._history.append(Message("user", "之前的消息"))
        func_call_agent._history.append(Message("assistant", "之前的回复"))

        messages = func_call_agent._build_messages("新消息")

        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "之前的消息"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "之前的回复"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "新消息"

    def test_build_messages_empty_system_prompt(self, mock_client):
        """测试空系统提示词"""
        agent = OpenAIFuncCallAgent(
            name="test_agent", client=mock_client, system_prompt=""
        )
        messages = agent._build_messages("测试")

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == ""


# ========== _build_tool_schemas 测试 ==========


class TestBuildToolSchemas:
    """测试 _build_tool_schemas 方法"""

    def test_build_schemas_without_tools(self, func_call_agent):
        """测试不带工具的 schema 构建"""
        schemas = func_call_agent._build_tool_schemas()

        assert schemas == []

    def test_build_schemas_with_tools(self, func_call_agent_with_tools, mock_tool):
        """测试带工具的 schema 构建"""
        schemas = func_call_agent_with_tools._build_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "test_tool"
        assert schemas[0]["function"]["description"] == "测试工具"
        assert "parameters" in schemas[0]["function"]
        assert schemas[0]["function"]["parameters"]["type"] == "object"

    def test_build_schemas_with_parameters(self, func_call_agent_with_tools):
        """测试工具参数 schema 构建"""
        schemas = func_call_agent_with_tools._build_tool_schemas()

        parameters = schemas[0]["function"]["parameters"]
        assert "properties" in parameters
        assert "param1" in parameters["properties"]
        assert "param2" in parameters["properties"]
        assert parameters["properties"]["param1"]["type"] == "string"
        assert parameters["properties"]["param2"]["type"] == "integer"

    def test_build_schemas_with_required_parameters(self, func_call_agent_with_tools):
        """测试必需参数标记"""
        schemas = func_call_agent_with_tools._build_tool_schemas()

        parameters = schemas[0]["function"]["parameters"]
        assert "required" in parameters
        assert "param1" in parameters["required"]

    def test_build_schemas_without_registry(self, mock_client):
        """测试空工具注册表"""
        agent = OpenAIFuncCallAgent(name="test_agent", client=mock_client)
        schemas = agent._build_tool_schemas()
        assert schemas == []

    def test_build_schemas_calls_to_schema(self, func_call_agent_with_tools, mock_tool):
        """测试调用工具的 to_schema 方法"""
        func_call_agent_with_tools._build_tool_schemas()

        # 验证调用了 to_schema 方法
        assert mock_tool.to_schema.called


# ========== _parse_function_call_arguments 测试 ==========


class TestParseFunctionCallArguments:
    """测试 _parse_function_call_arguments 方法"""

    def test_parse_valid_json(self, func_call_agent):
        """测试解析有效 JSON"""
        result = func_call_agent._parse_function_call_arguments('{"key": "value"}')

        assert result == {"key": "value"}

    def test_parse_empty_json(self, func_call_agent):
        """测试解析空 JSON"""
        result = func_call_agent._parse_function_call_arguments("{}")

        assert result == {}

    def test_parse_nested_json(self, func_call_agent):
        """测试解析嵌套 JSON"""
        json_str = '{"key1": "value1", "key2": {"nested": "value2"}}'
        result = func_call_agent._parse_function_call_arguments(json_str)

        assert result == {
            "key1": "value1",
            "key2": {"nested": "value2"},
        }

    def test_parse_invalid_json(self, func_call_agent):
        """测试解析无效 JSON"""
        result = func_call_agent._parse_function_call_arguments("{invalid}")

        assert result == {}

    def test_parse_none(self, func_call_agent):
        """测试解析 None"""
        result = func_call_agent._parse_function_call_arguments(None)

        assert result == {}

    def test_parse_empty_string(self, func_call_agent):
        """测试解析空字符串"""
        result = func_call_agent._parse_function_call_arguments("")

        assert result == {}

    def test_parse_non_dict_json(self, func_call_agent):
        """测试解析非字典 JSON"""
        result = func_call_agent._parse_function_call_arguments('["array", "values"]')

        assert result == {}


# ========== _convert_parameter_types 测试 ==========


class TestConvertParameterTypes:
    """测试 _convert_parameter_types 方法"""

    def test_convert_string_to_string(self, func_call_agent_with_tools):
        """测试字符串类型转换"""
        result = func_call_agent_with_tools._convert_parameter_types(
            "test_tool", {"param1": "value"}
        )

        assert result["param1"] == "value"

    def test_convert_string_to_int(self, func_call_agent_with_tools):
        """测试字符串转整数"""
        result = func_call_agent_with_tools._convert_parameter_types(
            "test_tool", {"param2": "123"}
        )

        assert result["param2"] == 123
        assert isinstance(result["param2"], int)

    def test_convert_int_to_int(self, func_call_agent_with_tools):
        """测试整数类型保持"""
        result = func_call_agent_with_tools._convert_parameter_types(
            "test_tool", {"param2": 456}
        )

        assert result["param2"] == 456

    def test_convert_string_to_float(self, func_call_agent_with_tools, mock_tool):
        """测试字符串转浮点数"""
        # 修改工具参数为 number 类型
        mock_tool.get_parameters = Mock(
            return_value=[
                ToolParameter(
                    name="num", type="number", description="数字", required=True
                )
            ]
        )

        result = func_call_agent_with_tools._convert_parameter_types(
            "test_tool", {"num": "3.14"}
        )

        assert result["num"] == 3.14
        assert isinstance(result["num"], float)

    def test_convert_string_to_boolean(self, func_call_agent_with_tools, mock_tool):
        """测试字符串转布尔值"""
        mock_tool.get_parameters = Mock(
            return_value=[
                ToolParameter(
                    name="flag", type="boolean", description="标志", required=True
                )
            ]
        )

        assert (
            func_call_agent_with_tools._convert_parameter_types(
                "test_tool", {"flag": "true"}
            )["flag"]
            is True
        )
        assert (
            func_call_agent_with_tools._convert_parameter_types(
                "test_tool", {"flag": "false"}
            )["flag"]
            is False
        )
        assert (
            func_call_agent_with_tools._convert_parameter_types(
                "test_tool", {"flag": "1"}
            )["flag"]
            is True
        )

    def test_convert_with_empty_registry(self, mock_client):
        """测试空工具注册表时的参数转换"""
        agent = OpenAIFuncCallAgent(name="test_agent", client=mock_client)
        result = agent._convert_parameter_types("test_tool", {"key": "value"})
        assert result == {"key": "value"}

    def test_convert_unknown_tool(self, func_call_agent_with_tools):
        """测试未知工具"""
        result = func_call_agent_with_tools._convert_parameter_types(
            "unknown_tool", {"key": "value"}
        )

        assert result == {"key": "value"}


# ========== _execute_tool_call 测试 ==========


class TestExecuteToolCall:
    """测试 _execute_tool_call 方法"""

    def test_execute_tool_success(self, func_call_agent_with_tools, mock_tool):
        """测试成功执行工具"""
        result = func_call_agent_with_tools._execute_tool_call(
            "test_tool", {"param1": "value1"}
        )

        assert "工具执行成功" in result
        mock_tool.run.assert_called_once()

    def test_execute_tool_not_found(self, func_call_agent_with_tools):
        """测试工具不存在时抛出 ValueError"""
        with pytest.raises(ValueError, match="未找到工具 'nonexistent_tool'"):
            func_call_agent_with_tools._execute_tool_call("nonexistent_tool", {})

    def test_execute_tool_without_registry(self, func_call_agent):
        """测试没有工具注册表时抛出 ValueError"""
        func_call_agent.tool_registry = None

        with pytest.raises(ValueError, match="未配置工具注册表"):
            func_call_agent._execute_tool_call("test_tool", {})

    def test_execute_tool_exception(self, func_call_agent_with_tools, mock_tool):
        """测试工具执行异常"""
        mock_tool.run = Mock(side_effect=Exception("工具错误"))

        result = func_call_agent_with_tools._execute_tool_call("test_tool", {})

        assert "工具调用失败" in result
        assert "工具错误" in result


# ========== run 方法测试 ==========


class TestRun:
    """测试 run 方法"""

    def test_run_without_tools(
        self, func_call_agent, mock_chat_completion_without_tools
    ):
        """测试不带工具的运行"""
        func_call_agent.client.chat_with_tools = Mock(
            return_value=mock_chat_completion_without_tools
        )

        response = func_call_agent.run("你好")

        assert response == "你好！有什么可以帮助你的吗？"
        assert len(func_call_agent._history) == 2
        assert func_call_agent._history[0].role == "user"
        assert func_call_agent._history[0].content == "你好"
        assert func_call_agent._history[1].role == "assistant"
        assert func_call_agent._history[1].content == response

    def test_run_with_single_tool_call(
        self,
        func_call_agent_with_tools,
        mock_chat_completion_with_tool_call,
        mock_tool,
    ):
        """测试带单个工具调用的运行"""
        # 创建一个没有工具调用的最终响应
        final_message = ChatCompletionMessage(
            role="assistant",
            content="工具执行完成,这是最终答案",
        )
        final_choice = Choice(index=0, message=final_message, finish_reason="stop")
        final_completion = ChatCompletion(
            id="test-id-final",
            created=1234567891,
            model="test-model",
            choices=[final_choice],
            object="chat.completion",
        )

        func_call_agent_with_tools.client.chat_with_tools = Mock(
            side_effect=[
                mock_chat_completion_with_tool_call,  # 第一次:工具调用
                final_completion,  # 第二次:最终响应
            ]
        )

        response = func_call_agent_with_tools.run("执行测试工具")

        assert response == "工具执行完成,这是最终答案"
        assert func_call_agent_with_tools.client.chat_with_tools.call_count == 2
        mock_tool.run.assert_called_once()

    def test_run_with_max_tool_iterations_override(
        self, func_call_agent_with_tools, mock_chat_completion_with_tool_call
    ):
        """测试覆盖最大工具迭代次数"""
        # 创建一个有 content 的最终响应
        final_message = ChatCompletionMessage(role="assistant", content="最终答案")
        final_choice = Choice(index=0, message=final_message, finish_reason="stop")
        final_completion = ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[final_choice],
            object="chat.completion",
        )

        func_call_agent_with_tools.client.chat_with_tools = Mock(
            side_effect=[
                mock_chat_completion_with_tool_call,
                mock_chat_completion_with_tool_call,
                final_completion,
            ]
        )

        _ = func_call_agent_with_tools.run("执行测试", max_tool_iterations=2)

        # 应该调用 3 次：2 次工具调用 + 1 次最终获取
        assert func_call_agent_with_tools.client.chat_with_tools.call_count == 3

    def test_run_with_custom_tool_choice(
        self, func_call_agent, mock_chat_completion_without_tools
    ):
        """测试自定义 tool_choice"""
        func_call_agent.client.chat_with_tools = Mock(
            return_value=mock_chat_completion_without_tools
        )

        func_call_agent.run("测试", tool_choice="none")

        # 检查是否传递了 tool_choice
        call_kwargs = func_call_agent.client.chat_with_tools.call_args[1]
        assert call_kwargs["tool_choice"] == "none"

    def test_run_passes_kwargs(
        self, func_call_agent, mock_chat_completion_without_tools
    ):
        """测试传递额外参数"""
        func_call_agent.client.chat_with_tools = Mock(
            return_value=mock_chat_completion_without_tools
        )

        func_call_agent.run("测试", temperature=0.5, max_tokens=100)

        # 检查是否传递了额外参数
        call_kwargs = func_call_agent.client.chat_with_tools.call_args[1]
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.5


# ========== 工具管理测试 ==========


class TestToolManagement:
    """测试工具管理方法"""

    def test_add_tool(self, func_call_agent, mock_tool):
        """测试添加工具"""
        func_call_agent.add_tool(mock_tool)

        assert "test_tool" in func_call_agent.list_tools()

    def test_remove_tool(self, func_call_agent_with_tools):
        """测试移除工具"""
        result = func_call_agent_with_tools.remove_tool("test_tool")

        assert result is True
        assert "test_tool" not in func_call_agent_with_tools.list_tools()

    def test_remove_nonexistent_tool(self, func_call_agent_with_tools):
        """测试移除不存在的工具"""
        result = func_call_agent_with_tools.remove_tool("nonexistent_tool")

        assert result is False

    def test_remove_tool_without_registry(self, func_call_agent):
        """测试没有工具注册表时移除工具"""
        func_call_agent.tool_registry = None

        result = func_call_agent.remove_tool("test_tool")

        assert result is False

    def test_list_tools(self, func_call_agent_with_tools):
        """测试列出工具"""
        tools = func_call_agent_with_tools.list_tools()

        assert "test_tool" in tools

    def test_list_tools_without_registry(self, func_call_agent):
        """测试没有工具注册表时列出工具"""
        func_call_agent.tool_registry = None

        tools = func_call_agent.list_tools()

        assert tools == []

    def test_has_tools(self, func_call_agent_with_tools):
        """测试是否有工具"""
        assert func_call_agent_with_tools.has_tools() is True

    def test_has_tools_without_registry(self, func_call_agent):
        """测试没有工具注册表时"""
        func_call_agent.tool_registry = None

        assert func_call_agent.has_tools() is False


# ========== stream_run 方法测试 ==========


class TestStreamRun:
    """测试 stream_run 方法"""

    def test_stream_run_yields_result(self, func_call_agent):
        """测试流式运行返回结果"""
        # Mock run 方法
        func_call_agent.run = Mock(return_value="测试回复")

        chunks = list(func_call_agent.stream_run("测试输入"))

        assert len(chunks) == 1
        assert chunks[0] == "测试回复"


# ========== 继承的方法测试 ==========


class TestInheritedMethods:
    """测试从 Agent 基类继承的方法"""

    def test_add_message(self, func_call_agent):
        """测试添加消息"""
        msg = Message(role="user", content="测试消息")
        func_call_agent.add_message(msg)

        assert len(func_call_agent._history) == 1
        assert func_call_agent._history[0] == msg

    def test_clear_history(self, func_call_agent):
        """测试清空历史记录"""
        func_call_agent._history.append(Message(role="user", content="测试"))
        func_call_agent.clear_history()

        assert len(func_call_agent._history) == 0

    def test_get_history(self, func_call_agent):
        """测试获取历史记录"""
        msg1 = Message(role="user", content="消息1")
        msg2 = Message(role="assistant", content="消息2")
        func_call_agent.add_message(msg1)
        func_call_agent.add_message(msg2)

        history = func_call_agent.get_history()

        assert len(history) == 2
        # 应该返回副本，而不是引用
        assert history is not func_call_agent._history

    def test_str_representation(self, func_call_agent):
        """测试字符串表示"""
        str_repr = str(func_call_agent)

        assert "test_agent" in str_repr


# ========== 边界情况测试 ==========


class TestEdgeCases:
    """测试边界情况"""

    def test_run_with_empty_input(
        self, func_call_agent, mock_chat_completion_without_tools
    ):
        """测试空输入"""
        func_call_agent.client.chat_with_tools = Mock(
            return_value=mock_chat_completion_without_tools
        )

        response = func_call_agent.run("")

        assert response is not None

    def test_convert_parameter_with_invalid_value(self, func_call_agent_with_tools):
        """测试转换无效参数值"""
        # 当转换失败时，应该保留原值
        result = func_call_agent_with_tools._convert_parameter_types(
            "test_tool", {"param2": "invalid_int"}
        )

        # 由于 "invalid_int" 无法转换为 int，应该保留原值
        assert result["param2"] == "invalid_int"

    def test_build_tool_schemas_exception_handling(self, func_call_agent_with_tools):
        """测试工具参数获取异常处理"""
        # 创建一个会抛出异常的工具
        from liagents.tools.base import Tool as ToolClass

        def broken_func(param: str) -> str:
            return "test"

        class BrokenTool(ToolClass):
            def get_parameters(self):
                raise Exception("参数获取失败")

        broken_tool = BrokenTool(
            func=broken_func,
            name="broken",
            description="破损工具",
            parameters=[],  # 传入空列表，实际会通过 get_parameters 抛出异常
        )

        # 由于 get_parameters 会抛出异常，这个测试验证代码不会崩溃
        # to_schema 会调用 get_parameters，所以会抛出异常
        with pytest.raises(Exception, match="参数获取失败"):
            broken_tool.to_schema()

    def test_run_with_zero_max_iterations(
        self, func_call_agent_with_tools, mock_chat_completion_with_tool_call
    ):
        """测试最大迭代次数为 0"""
        # 当 max_tool_iterations=0 时，while 循环不会执行
        # 但由于 final_response 为空字符串，会调用 tool_choice="none" 获取最终答案
        final_message = ChatCompletionMessage(role="assistant", content="最终答案")
        final_choice = Choice(index=0, message=final_message, finish_reason="stop")
        final_completion = ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[final_choice],
            object="chat.completion",
        )

        func_call_agent_with_tools.client.chat_with_tools = Mock(
            return_value=final_completion
        )

        _ = func_call_agent_with_tools.run("测试", max_tool_iterations=0)

        # 由于循环不执行且 final_response 为空，会调用一次获取最终答案
        assert func_call_agent_with_tools.client.chat_with_tools.call_count == 1
