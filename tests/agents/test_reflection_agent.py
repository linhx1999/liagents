"""测试 ReflectionAgent"""

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

from liagents.agents.reflection_agent import ReflectionAgent
from liagents.core.client import Client
from liagents.core.config import Config
from liagents.core.message import Message
from liagents.tools.registry import tool_registry


# ========== 测试 Fixtures ==========


@pytest.fixture(autouse=True)
def cleanup_tool_registry():
    """每个测试后清理工具注册表"""
    yield
    tool_registry.clear()


@pytest.fixture
def mock_client():
    """创建 Mock Client 实例"""
    client = Mock(spec=Client)
    client.chat_with_tools = Mock()
    return client


@pytest.fixture
def mock_config():
    """创建 Mock Config 实例"""
    return Config()


@pytest.fixture
def reflection_agent(mock_client):
    """创建 ReflectionAgent 实例（使用默认配置）"""
    agent = ReflectionAgent(
        name="test_reflection_agent",
        client=mock_client,
        system_prompt="你是一个反思型测试助手",
    )
    yield agent
    # 测试后清理
    agent.tool_registry.clear()


@pytest.fixture
def reflection_agent_with_custom_prompt(mock_client):
    """创建带自定义系统提示词的 ReflectionAgent 实例"""
    return ReflectionAgent(
        name="custom_reflection_agent",
        client=mock_client,
        system_prompt="自定义反思提示词",
    )


@pytest.fixture
def mock_chat_completion_without_tools():
    """创建不带工具调用的模拟响应"""
    message = ChatCompletionMessage(
        role="assistant",
        content="反思完成，这是我的结论。",
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
def mock_chat_completion_with_think_call():
    """创建带 think 工具调用的模拟响应"""
    tool_call = ChatCompletionMessageToolCall(
        id="call_123",
        type="function",
        function=Function(
            name="think",
            arguments='{"thought": "让我思考一下这个问题的解决方案..."}',
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


class TestReflectionAgentInit:
    """测试 ReflectionAgent 初始化"""

    def test_init_basic(self, mock_client):
        """测试基本初始化"""
        agent = ReflectionAgent(
            name="test_agent",
            client=mock_client,
            system_prompt="测试提示词",
        )

        assert agent.name == "test_agent"
        assert agent.client == mock_client
        assert agent.system_prompt == "测试提示词"
        assert agent.tool_registry is not None
        assert agent._history == []

    def test_init_without_name(self, mock_client):
        """测试默认名称"""
        agent = ReflectionAgent(client=mock_client)

        assert agent.name == "ReflectionAgent"

    def test_init_with_custom_max_iterations(self, mock_client):
        """测试自定义 max_tool_iterations"""
        agent = ReflectionAgent(
            name="test_agent",
            client=mock_client,
            max_tool_iterations=5,
        )

        assert agent.max_tool_iterations == 5

    def test_init_with_default_tool_choice(self, mock_client):
        """测试默认 tool_choice"""
        agent = ReflectionAgent(name="test_agent", client=mock_client)

        assert agent.tool_choice == "auto"


# ========== 默认工具测试 ==========


class TestDefaultTools:
    """测试默认工具配置"""

    def test_think_tool_registered(self, reflection_agent):
        """测试 think 工具已注册"""
        tools = reflection_agent.list_tools()

        assert "think" in tools

    def test_has_think_tool(self, reflection_agent):
        """测试有 think 工具"""
        assert reflection_agent.has_tools() is True

    def test_get_think_tool(self, reflection_agent):
        """测试获取 think 工具"""
        tool = reflection_agent.tool_registry.get_tool("think")

        assert tool is not None
        assert tool.name == "think"

    def test_can_remove_think_tool(self, reflection_agent):
        """测试可以移除 think 工具"""
        result = reflection_agent.remove_tool("think")

        assert result is True
        assert "think" not in reflection_agent.list_tools()

    def test_only_think_tool_by_default(self, reflection_agent):
        """测试默认只有 think 工具"""
        tools = reflection_agent.list_tools()

        assert len(tools) == 1
        assert "think" in tools
        assert "write_todos" not in tools


# ========== 继承方法测试 ==========


class TestInheritedMethods:
    """测试从 OpenAIFuncCallAgent 继承的方法"""

    def test_add_tool(self, reflection_agent):
        """测试添加工具"""
        from liagents.tools.base import Tool

        mock_tool = Mock(spec=Tool)
        mock_tool.name = "extra_tool"
        mock_tool.description = "额外工具"

        reflection_agent.add_tool(mock_tool)

        assert "extra_tool" in reflection_agent.list_tools()

    def test_list_tools_includes_think(self, reflection_agent):
        """测试列出工具包含 think"""
        tools = reflection_agent.list_tools()

        assert len(tools) >= 1
        assert "think" in tools

    def test_run_inherits_from_parent(
        self, reflection_agent, mock_chat_completion_without_tools
    ):
        """测试 run 方法继承正常"""
        reflection_agent.client.chat_with_tools = Mock(
            return_value=mock_chat_completion_without_tools
        )

        response = reflection_agent.run("测试任务")

        assert response == "反思完成，这是我的结论。"

    def test_run_with_history(self, reflection_agent):
        """测试带历史记录的运行"""
        reflection_agent._history.append(Message("user", "之前的消息"))
        reflection_agent._history.append(Message("assistant", "之前的回复"))

        mock_message = ChatCompletionMessage(role="assistant", content="新的反思")
        mock_choice = Choice(index=0, message=mock_message, finish_reason="stop")
        mock_completion = ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[mock_choice],
            object="chat.completion",
        )
        reflection_agent.client.chat_with_tools.return_value = mock_completion

        response = reflection_agent.run("新消息")

        # 验证响应正确
        assert response == "新的反思"

        # 验证历史记录更新：原有2条 + user消息 + assistant响应 = 4条
        assert len(reflection_agent._history) == 4

    def test_stream_run_inherits_from_parent(self, reflection_agent):
        """测试 stream_run 方法继承正常"""
        reflection_agent.run = Mock(return_value="流式反思回复")

        chunks = list(reflection_agent.stream_run("测试输入"))

        assert len(chunks) == 1
        assert chunks[0] == "流式反思回复"


# ========== 运行测试 ==========


class TestRun:
    """测试 run 方法"""

    def test_run_without_tools(
        self, reflection_agent, mock_chat_completion_without_tools
    ):
        """测试不带工具的运行"""
        reflection_agent.client.chat_with_tools = Mock(
            return_value=mock_chat_completion_without_tools
        )

        response = reflection_agent.run("你好")

        assert response == "反思完成，这是我的结论。"
        assert len(reflection_agent._history) == 2

    def test_run_with_tool_call(
        self, reflection_agent, mock_chat_completion_with_think_call
    ):
        """测试带工具调用的运行"""
        # 最终响应
        final_message = ChatCompletionMessage(role="assistant", content="经过思考，这是我的答案")
        final_choice = Choice(index=0, message=final_message, finish_reason="stop")
        final_completion = ChatCompletion(
            id="test-id-final",
            created=1234567891,
            model="test-model",
            choices=[final_choice],
            object="chat.completion",
        )

        reflection_agent.client.chat_with_tools = Mock(
            side_effect=[
                mock_chat_completion_with_think_call,  # 第一次:工具调用
                final_completion,  # 第二次:最终响应
            ]
        )

        response = reflection_agent.run("分析这个问题")

        assert response == "经过思考，这是我的答案"
        assert reflection_agent.client.chat_with_tools.call_count == 2

    def test_run_with_custom_max_iterations(self, reflection_agent):
        """测试自定义 max_tool_iterations"""
        mock_message = ChatCompletionMessage(role="assistant", content="反思完成！")
        mock_choice = Choice(index=0, message=mock_message, finish_reason="stop")
        mock_completion = ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[mock_choice],
            object="chat.completion",
        )
        reflection_agent.client.chat_with_tools.return_value = mock_completion

        response = reflection_agent.run("测试", max_tool_iterations=5)

        assert response == "反思完成！"

    def test_run_passes_kwargs(
        self, reflection_agent, mock_chat_completion_without_tools
    ):
        """测试传递额外参数"""
        reflection_agent.client.chat_with_tools = Mock(
            return_value=mock_chat_completion_without_tools
        )

        reflection_agent.run("测试", temperature=0.5, max_tokens=100)

        call_kwargs = reflection_agent.client.chat_with_tools.call_args[1]
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.5


# ========== 边界情况测试 ==========


class TestEdgeCases:
    """测试边界情况"""

    def test_run_with_empty_input(self, reflection_agent):
        """测试空输入"""
        mock_message = ChatCompletionMessage(role="assistant", content="")
        mock_choice = Choice(index=0, message=mock_message, finish_reason="stop")
        mock_completion = ChatCompletion(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[mock_choice],
            object="chat.completion",
        )
        reflection_agent.client.chat_with_tools = Mock(return_value=mock_completion)

        response = reflection_agent.run("")

        assert response == ""

    def test_add_multiple_tools(self, reflection_agent):
        """测试添加多个工具"""
        from liagents.tools.base import Tool

        mock_tool1 = Mock(spec=Tool)
        mock_tool1.name = "tool1"
        mock_tool1.description = "工具1"

        mock_tool2 = Mock(spec=Tool)
        mock_tool2.name = "tool2"
        mock_tool2.description = "工具2"

        reflection_agent.add_tool(mock_tool1)
        reflection_agent.add_tool(mock_tool2)

        tools = reflection_agent.list_tools()
        assert "tool1" in tools
        assert "tool2" in tools
        assert "think" in tools  # 默认工具应仍存在

    def test_remove_all_tools(self, reflection_agent):
        """测试移除所有工具"""
        reflection_agent.remove_tool("think")

        # 检查 list_tools() 是否为空
        assert reflection_agent.list_tools() == []
        assert len(reflection_agent.tool_registry.list_tools()) == 0

    def test_reflection_agent_vs_plan_solve_agent(self, mock_client):
        """测试 ReflectionAgent 比 PlanSolveAgent 少 write_todos 工具"""
        from liagents.agents.plan_solve_agent import PlanSolveAgent

        reflection_agent = ReflectionAgent(client=mock_client)
        plan_solve_agent = PlanSolveAgent(client=mock_client)

        reflection_tools = reflection_agent.list_tools()
        plan_solve_tools = plan_solve_agent.list_tools()

        # ReflectionAgent 只有 think
        assert "think" in reflection_tools
        assert "write_todos" not in reflection_tools

        # PlanSolveAgent 有 think 和 write_todos
        assert "think" in plan_solve_tools
        assert "write_todos" in plan_solve_tools
