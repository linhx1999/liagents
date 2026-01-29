"""测试 Agent 基类"""

import pytest
from unittest.mock import Mock, patch as mock_patch
import os

# 设置必需的环境变量
os.environ["MODEL"] = "test-model"
os.environ["OPENAI_API_KEY"] = "test-api-key"
os.environ["OPENAI_BASE_URL"] = "https://test.example.com"

from liagents.core.agent import Agent
from liagents.core.client import Client
from liagents.core.config import Config
from liagents.core.message import Message
from liagents.tools.registry import ToolRegistry


# 创建具体的 Agent 子类用于测试
class ConcreteAgent(Agent):
    """用于测试的具体 Agent 实现"""

    def run(self, input_text: str, **kwargs) -> str:
        """实现抽象方法"""
        return f"Processed: {input_text}"


class TestAgentInit:
    """测试 Agent 初始化"""

    def test_init_basic(self):
        """测试基本初始化"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(
            name="test_agent",
            client=client,
            system_prompt="测试提示词",
        )

        assert agent.name == "test_agent"
        assert agent.client == client
        assert agent.system_prompt == "测试提示词"
        assert agent.config is not None
        assert agent._history == []
        assert agent.tool_registry is not None
        assert agent.debug is False

    def test_init_with_config(self):
        """测试带配置的初始化"""
        client = Mock(spec=Client)
        config = Config(model="test-model", temperature=0.3)
        agent = ConcreteAgent(name="test_agent", client=client, config=config)

        assert agent.config == config
        assert agent.config.temperature == 0.3

    def test_init_with_tool_registry(self):
        """测试带工具注册表的初始化"""
        client = Mock(spec=Client)
        registry = ToolRegistry()
        agent = ConcreteAgent(name="test_agent", client=client, tool_registry=registry)

        assert agent.tool_registry == registry

    def test_init_with_debug(self):
        """测试带调试模式的初始化"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client, debug=True)

        assert agent.debug is True

    def test_init_default_values(self):
        """测试默认值"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        assert agent.system_prompt == ""
        assert agent._history == []
        assert agent.tool_registry is not None


class TestAgentDebugPrint:
    """测试调试打印方法"""

    def test_debug_print_disabled(self):
        """测试调试打印关闭时"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client, debug=False)

        # 不应该输出任何内容
        result = agent._debug_print("Test", {"key": "value"})
        assert result is None

    def test_debug_print_enabled(self):
        """测试调试打印开启时"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client, debug=True)

        # 应该输出内容（通过 mock print 捕获）
        with mock_patch("builtins.print") as mock_print:
            agent._debug_print("Test", {"key": "value"})
            mock_print.assert_called_once()

    def test_debug_print_dict_formatting(self):
        """测试字典格式化"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client, debug=True)

        with mock_patch("builtins.print") as mock_print:
            agent._debug_print("Test", {"key": "value"})
            call_args = mock_print.call_args[0][0]
            assert "Test" in call_args
            assert "key" in call_args

    def test_debug_print_string_content(self):
        """测试字符串内容"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client, debug=True)

        with mock_patch("builtins.print") as mock_print:
            agent._debug_print("Test", "simple string")
            call_args = mock_print.call_args[0][0]
            assert "Test" in call_args
            assert "simple string" in call_args


class TestAgentHistory:
    """测试历史记录管理"""

    def test_add_message(self):
        """测试添加消息"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        msg = Message(role="user", content="测试消息")
        agent.add_message(msg)

        assert len(agent._history) == 1
        assert agent._history[0] == msg

    def test_clear_history(self):
        """测试清空历史记录"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        agent._history.append(Message(role="user", content="消息1"))
        agent._history.append(Message(role="assistant", content="消息2"))

        agent.clear_history()

        assert len(agent._history) == 0

    def test_get_history(self):
        """测试获取历史记录"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        agent._history.append(Message(role="user", content="消息1"))
        agent._history.append(Message(role="assistant", content="消息2"))

        history = agent.get_history()

        assert len(history) == 2
        # 应该返回副本，不是引用
        assert history is not agent._history

    def test_get_history_empty(self):
        """测试获取空历史记录"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        history = agent.get_history()

        assert len(history) == 0
        assert history == []


class TestAgentToolManagement:
    """测试工具管理"""

    def test_add_tool(self):
        """测试添加工具"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        mock_tool = Mock()
        mock_tool.name = "test_tool"

        # Mock register_tool 方法
        with mock_patch.object(agent.tool_registry, "register_tool") as mock_register:
            agent.add_tool(mock_tool)
            mock_register.assert_called_once_with(mock_tool)

    def test_remove_tool_success(self):
        """测试移除工具成功"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        agent.tool_registry.unregister_tool = Mock(return_value=True)
        result = agent.remove_tool("test_tool")

        assert result is True
        agent.tool_registry.unregister_tool.assert_called_once_with("test_tool")

    def test_remove_tool_not_found(self):
        """测试移除不存在的工具"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        agent.tool_registry.unregister_tool = Mock(return_value=False)
        result = agent.remove_tool("nonexistent")

        assert result is False

    def test_remove_tool_no_registry(self):
        """测试没有工具注册表时移除工具"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)
        agent.tool_registry = None

        result = agent.remove_tool("test_tool")

        assert result is False

    def test_list_tools(self):
        """测试列出工具"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        agent.tool_registry.list_tools = Mock(return_value=["tool1", "tool2"])
        tools = agent.list_tools()

        assert tools == ["tool1", "tool2"]
        agent.tool_registry.list_tools.assert_called_once()

    def test_list_tools_no_registry(self):
        """测试没有工具注册表时列出工具"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)
        agent.tool_registry = None

        tools = agent.list_tools()

        assert tools == []

    def test_has_tools_true(self):
        """测试有工具"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        assert agent.has_tools() is True

    def test_has_tools_no_registry(self):
        """测试没有工具注册表时"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)
        agent.tool_registry = None

        assert agent.has_tools() is False


class TestAgentStr:
    """测试 Agent 字符串表示"""

    def test_str_representation(self):
        """测试字符串表示"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        result = str(agent)
        assert result == "Agent(name=test_agent)"


class TestAgentRun:
    """测试 Agent run 方法"""

    def test_run_returns_result(self):
        """测试 run 方法返回结果"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        result = agent.run("输入文本")

        assert result == "Processed: 输入文本"

    def test_run_with_kwargs(self):
        """测试 run 方法传递额外参数"""
        client = Mock(spec=Client)
        agent = ConcreteAgent(name="test_agent", client=client)

        result = agent.run("输入", temperature=0.5)

        assert result == "Processed: 输入"
