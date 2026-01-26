import json
import pytest
from unittest.mock import Mock

from liagents.agents.react_agent import ReActAgent
from liagents.core.client import Client
from liagents.core.config import Config
from liagents.core.message import Message
from liagents.tools.registry import ToolRegistry
from liagents.tools.base import Tool


# ========== 测试 Fixtures ==========


@pytest.fixture
def mock_client():
    """创建 Mock Client 实例"""
    client = Mock(spec=Client)
    client.invoke_chat = Mock(return_value="测试响应")
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
    tool.get_parameters = Mock(return_value=[])
    return tool


@pytest.fixture
def empty_tool_registry():
    """创建空的 Mock ToolRegistry 实例"""
    registry = Mock(spec=ToolRegistry)
    registry.get_tools_description = Mock(return_value="暂无可用工具")
    registry.get_tool = Mock(return_value=None)
    registry.list_tools = Mock(return_value=[])
    registry.unregister_tool = Mock(return_value=False)
    registry.register_tool = Mock()
    return registry


@pytest.fixture
def mock_tool_registry(mock_tool):
    """创建包含工具的 Mock ToolRegistry 实例"""
    registry = Mock(spec=ToolRegistry)
    registry.get_tools_description = Mock(
        return_value=f"- {mock_tool.name}: {mock_tool.description}"
    )
    registry.get_tool = Mock(return_value=mock_tool)
    registry.list_tools = Mock(return_value=[mock_tool.name])
    registry.unregister_tool = Mock(return_value=True)
    registry.register_tool = Mock()
    return registry


@pytest.fixture
def react_agent(mock_client, mock_config, empty_tool_registry):
    """创建 ReActAgent 实例（不带工具）"""
    return ReActAgent(
        name="test_agent",
        client=mock_client,
        system_prompt="你是一个测试助手",
        config=mock_config,
        tool_registry=empty_tool_registry,
    )


@pytest.fixture
def react_agent_with_tools(mock_client, mock_config, mock_tool_registry):
    """创建 ReActAgent 实例（带工具）"""
    return ReActAgent(
        name="test_agent_with_tools",
        client=mock_client,
        system_prompt="你是一个测试助手",
        config=mock_config,
        tool_registry=mock_tool_registry,
    )


# ========== 初始化测试 ==========


class TestReActAgentInit:
    """测试 ReActAgent 初始化"""

    def test_init_basic(self, mock_client, mock_config):
        """测试基本初始化"""
        agent = ReActAgent(
            name="test_agent",
            client=mock_client,
            system_prompt="测试提示词",
            config=mock_config,
        )

        assert agent.name == "test_agent"
        assert agent.client == mock_client
        assert agent.system_prompt == "测试提示词"
        assert agent.config == mock_config
        assert agent.tool_registry is not None
        assert agent._history == []

    def test_init_without_system_prompt(self, mock_client, mock_config):
        """测试不提供系统提示词"""
        agent = ReActAgent(name="test_agent", client=mock_client, config=mock_config)

        assert agent.system_prompt == ""

    def test_init_with_tool_registry(self, mock_client, mock_tool_registry):
        """测试带工具注册表的初始化"""
        agent = ReActAgent(
            name="test_agent", client=mock_client, tool_registry=mock_tool_registry
        )

        assert agent.tool_registry == mock_tool_registry

    def test_init_creates_default_tool_registry(self, mock_client):
        """测试默认创建 ToolRegistry"""
        agent = ReActAgent(name="test_agent", client=mock_client)

        assert agent.tool_registry is not None


# ========== _get_enhanced_system_prompt 测试 ==========


class TestGetEnhancedSystemPrompt:
    """测试 _get_enhanced_system_prompt 方法"""

    def test_without_tools(self, react_agent):
        """测试不带工具时的系统提示词"""
        prompt = react_agent._get_enhanced_system_prompt()

        assert prompt == "你是一个测试助手"

    def test_without_system_prompt_and_tools(self, mock_client, mock_config):
        """测试没有系统提示词和工具时的默认提示词"""
        agent = ReActAgent(name="test_agent", client=mock_client, config=mock_config)
        prompt = agent._get_enhanced_system_prompt()

        assert prompt == ""

    def test_with_tools(self, react_agent_with_tools):
        """测试带工具时的系统提示词"""
        prompt = react_agent_with_tools._get_enhanced_system_prompt()

        assert "你是一个测试助手" in prompt
        assert "# Tools" in prompt
        assert "test_tool" in prompt
        assert "测试工具" in prompt
        assert "<tools>" in prompt
        assert "</tools>" in prompt
        assert "tool_call" in prompt

    def test_with_empty_tool_registry(self, mock_client):
        """测试空工具注册表"""
        empty_registry = ToolRegistry()
        agent = ReActAgent(
            name="test_agent",
            client=mock_client,
            system_prompt="测试助手",
            tool_registry=empty_registry,
        )

        prompt = agent._get_enhanced_system_prompt()

        # 空工具注册表应该返回基础提示词
        assert prompt == "测试助手"


# ========== _parse_tool_calls 测试 ==========


class TestParseToolCalls:
    """测试 _parse_tool_calls 方法"""

    def test_parse_single_tool_call(self, react_agent):
        """测试解析单个工具调用"""
        text = """让我调用这个工具：
<tool_call>
{"name": "calculator", "arguments": {"expression": "2+2"}}
</tool_call>"""

        tool_calls = react_agent._parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["tool_name"] == "calculator"
        assert tool_calls[0]["arguments"] == {"expression": "2+2"}

    def test_parse_multiple_tool_calls(self, react_agent):
        """测试解析多个工具调用"""
        text = """调用第一个工具：
<tool_call>
{"name": "calculator", "arguments": {"expression": "2+2"}}
</tool_call>

调用第二个工具：
<tool_call>
{"name": "search", "arguments": {"query": "Python"}}
</tool_call>"""

        tool_calls = react_agent._parse_tool_calls(text)

        assert len(tool_calls) == 2
        assert tool_calls[0]["tool_name"] == "calculator"
        assert tool_calls[1]["tool_name"] == "search"

    def test_parse_no_tool_calls(self, react_agent):
        """测试没有工具调用的情况"""
        text = "这是一个普通的回答，没有工具调用"

        tool_calls = react_agent._parse_tool_calls(text)

        assert len(tool_calls) == 0

    def test_parse_with_extra_text(self, react_agent):
        """测试工具调用在文本中间的情况"""
        text = """让我帮你计算一下
<tool_call>
{"name": "calculator", "arguments": {"expression": "2+2"}}
</tool_call>
结果是 4"""

        tool_calls = react_agent._parse_tool_calls(text)

        assert len(tool_calls) == 1
        assert tool_calls[0]["tool_name"] == "calculator"


# ========== _execute_tool_call 测试 ==========


class TestExecuteToolCall:
    """测试 _execute_tool_call 方法"""

    def test_execute_tool_success(self, react_agent_with_tools, mock_tool):
        """测试成功执行工具"""
        result = react_agent_with_tools._execute_tool_call(
            "test_tool", {"param1": "value1"}
        )

        assert "test_tool" in result
        assert "工具执行成功" in result
        mock_tool.run.assert_called_once_with({"param1": "value1"})

    def test_execute_tool_not_found(self, react_agent_with_tools):
        """测试工具不存在的情况"""
        react_agent_with_tools.tool_registry.get_tool = Mock(return_value=None)

        result = react_agent_with_tools._execute_tool_call("nonexistent_tool", {})

        assert "未找到工具" in result
        assert "nonexistent_tool" in result

    def test_execute_tool_without_registry(self, react_agent):
        """测试没有工具注册表的情况"""
        react_agent.tool_registry = None

        result = react_agent._execute_tool_call("test_tool", {})

        assert "未配置工具注册表" in result

    def test_execute_tool_exception(self, react_agent_with_tools, mock_tool):
        """测试工具执行异常的情况"""
        mock_tool.run = Mock(side_effect=Exception("工具错误"))

        result = react_agent_with_tools._execute_tool_call("test_tool", {})

        assert "工具调用失败" in result
        assert "工具错误" in result


# ========== run 方法测试 ==========


class TestRun:
    """测试 run 方法"""

    def test_run_without_tools(self, react_agent):
        """测试不带工具的运行"""
        react_agent.client.invoke_chat = Mock(return_value="你好！有什么可以帮助你的吗？")

        response = react_agent.run("你好")

        assert response == "你好！有什么可以帮助你的吗？"
        assert len(react_agent._history) == 2
        assert react_agent._history[0].role == "user"
        assert react_agent._history[0].content == "你好"

    def test_run_with_tool_call(self, react_agent_with_tools, mock_tool):
        """测试带工具调用的运行"""
        # 第一次返回工具调用，第二次返回最终答案
        react_agent_with_tools.client.invoke_chat = Mock(
            side_effect=[
                '<tool_call>\n{"name": "test_tool", "arguments": {"param": "value"}}\n</tool_call>',
                "工具执行完成，结果是：成功",
            ]
        )

        response = react_agent_with_tools.run("执行测试工具")

        assert response == "工具执行完成，结果是：成功"
        assert react_agent_with_tools.client.invoke_chat.call_count == 2
        mock_tool.run.assert_called_once()

    def test_run_with_max_iterations_reached(self, react_agent_with_tools):
        """测试达到最大迭代次数"""
        # 持续返回工具调用
        react_agent_with_tools.client.invoke_chat = Mock(
            return_value='<tool_call>\n{"name": "test_tool", "arguments": {}}\n</tool_call>'
        )

        _ = react_agent_with_tools.run("执行测试", max_tool_iterations=2)

        # 应该在最后一次获取最终答案
        assert react_agent_with_tools.client.invoke_chat.call_count == 3

    def test_run_with_history(self, react_agent):
        """测试带历史记录的运行"""
        react_agent._history.append(Message("user", "之前的消息"))
        react_agent._history.append(Message("assistant", "之前的回复"))

        react_agent.client.invoke_chat = Mock(return_value="新的回复")

        response = react_agent.run("新消息")

        # 检查是否传递了历史消息
        call_args = react_agent.client.invoke_chat.call_args[0][0]
        assert len(call_args) == 4  # system + 2 history + 1 new user
        assert response == "新的回复"

    def test_run_passes_kwargs(self, react_agent):
        """测试传递额外参数"""
        react_agent.client.invoke_chat = Mock(return_value="响应")

        react_agent.run("测试", temperature=0.5, max_tokens=100)

        # 检查是否传递了额外参数
        react_agent.client.invoke_chat.assert_called_once()
        call_kwargs = react_agent.client.invoke_chat.call_args[1]
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.5


# ========== 工具管理测试 ==========


class TestToolManagement:
    """测试工具管理方法"""

    def test_add_tool(self, react_agent, mock_tool):
        """测试添加工具"""
        react_agent.add_tool(mock_tool)

        react_agent.tool_registry.register_tool.assert_called_once_with(mock_tool)

    def test_remove_tool(self, react_agent_with_tools):
        """测试移除工具"""
        result = react_agent_with_tools.remove_tool("test_tool")

        react_agent_with_tools.tool_registry.unregister_tool.assert_called_once_with(
            "test_tool"
        )
        assert result is True

    def test_remove_tool_without_registry(self, react_agent):
        """测试没有工具注册表时移除工具"""
        react_agent.tool_registry = None

        result = react_agent.remove_tool("test_tool")

        assert result is False

    def test_list_tools(self, react_agent_with_tools):
        """测试列出工具"""
        tools = react_agent_with_tools.list_tools()

        react_agent_with_tools.tool_registry.list_tools.assert_called_once()
        assert tools == ["test_tool"]

    def test_list_tools_without_registry(self, react_agent):
        """测试没有工具注册表时列出工具"""
        react_agent.tool_registry = None

        tools = react_agent.list_tools()

        assert tools == []


# ========== stream_run 方法测试 ==========


class TestStreamRun:
    """测试 stream_run 方法"""

    def test_stream_run_yields_chunks(self, react_agent):
        """测试流式运行返回片段"""
        # Mock client 的 stream_chat 方法
        react_agent.client.stream_chat = Mock(
            return_value=iter(["你", "好", "！", "有", "什", "么", "帮", "助"])
        )

        chunks = list(react_agent.stream_run("你好"))

        assert len(chunks) == 8
        assert chunks == ["你", "好", "！", "有", "什", "么", "帮", "助"]

    def test_stream_run_saves_to_history(self, react_agent):
        """测试流式运行保存到历史记录"""
        # Mock client 的 stream_chat 方法
        react_agent.client.stream_chat = Mock(return_value=iter(["测试", "回复"]))

        list(react_agent.stream_run("测试输入"))

        # 检查历史记录
        assert len(react_agent._history) == 2
        assert react_agent._history[0].content == "测试输入"
        assert react_agent._history[1].content == "测试回复"


# ========== 继承的方法测试 ==========


class TestInheritedMethods:
    """测试从 Agent 基类继承的方法"""

    def test_add_message(self, react_agent):
        """测试添加消息"""
        msg = Message(role="user", content="测试消息")
        react_agent.add_message(msg)

        assert len(react_agent._history) == 1
        assert react_agent._history[0] == msg

    def test_clear_history(self, react_agent):
        """测试清空历史记录"""
        react_agent._history.append(Message(role="user", content="测试"))
        react_agent.clear_history()

        assert len(react_agent._history) == 0

    def test_get_history(self, react_agent):
        """测试获取历史记录"""
        msg1 = Message(role="user", content="消息1")
        msg2 = Message(role="assistant", content="消息2")
        react_agent.add_message(msg1)
        react_agent.add_message(msg2)

        history = react_agent.get_history()

        assert len(history) == 2
        # 应该返回副本，而不是引用
        assert history is not react_agent._history

    def test_str_representation(self, react_agent):
        """测试字符串表示"""
        str_repr = str(react_agent)

        assert "test_agent" in str_repr


# ========== 边界情况测试 ==========


class TestEdgeCases:
    """测试边界情况"""

    def test_run_with_empty_input(self, react_agent):
        """测试空输入"""
        react_agent.client.invoke_chat = Mock(return_value="")

        response = react_agent.run("")

        assert response == ""

    def test_parse_malformed_tool_call(self, react_agent):
        """测试格式错误的工具调用"""
        text = "<tool_call>\n{invalid json}\n</tool_call>"

        # 应该抛出 JSON 解析错误
        with pytest.raises(json.JSONDecodeError):
            react_agent._parse_tool_calls(text)

    def test_run_with_special_characters_in_arguments(self, react_agent_with_tools):
        """测试参数中包含特殊字符"""
        react_agent_with_tools.client.invoke_chat = Mock(
            return_value='{"name": "test_tool", "arguments": {"text": "Hello\\nWorld\\t!"}}'
        )

        response = react_agent_with_tools.run("测试")

        # 应该正常处理特殊字符
        assert isinstance(response, str)
        # 修复未使用变量警告
        assert response is not None  # noqa: F841
