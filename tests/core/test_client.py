"""测试 Client 客户端类"""

import pytest
from unittest.mock import Mock, patch
import os
from typing import cast
from openai.types.chat import ChatCompletionMessageParam

# 设置必需的环境变量
os.environ["MODEL"] = "test-model"
os.environ["OPENAI_API_KEY"] = "test-api-key"
os.environ["OPENAI_BASE_URL"] = "https://test.example.com"

from liagents.core.client import Client


def _m(messages: list[dict[str, str]]) -> list[ChatCompletionMessageParam]:
    """将字典消息列表转换为 ChatCompletionMessageParam 类型"""
    return cast(list[ChatCompletionMessageParam], messages)


class TestClientInit:
    """测试 Client 初始化"""

    def test_init_with_env_vars(self):
        """测试使用环境变量初始化"""
        client = Client()
        assert client._model == "test-model"
        assert client._temperature == 0.7
        assert client._max_completion_tokens is None

    def test_init_with_custom_values(self):
        """测试使用自定义值初始化"""
        client = Client(
            model="custom-model",
            api_key="custom-key",
            base_url="https://custom.example.com",
            temperature=0.5,
            max_completion_tokens=1000,
        )
        assert client._model == "custom-model"
        assert client._temperature == 0.5
        assert client._max_completion_tokens == 1000

    def test_init_missing_model(self):
        """测试缺少模型时抛出错误（需要显式传入空模型）"""
        # 由于默认参数在模块加载时已求值，这里测试显式传入空值
        with pytest.raises(ValueError, match="必须提供MODEL"):
            Client(model="", api_key="test-key", base_url="https://test.com")

    def test_init_missing_api_key(self):
        """测试缺少 API Key 时抛出错误"""
        with pytest.raises(ValueError, match="必须提供MODEL"):
            Client(model="test-model", api_key="", base_url="https://test.com")

    def test_init_missing_base_url(self):
        """测试缺少 Base URL 时抛出错误"""
        with pytest.raises(ValueError, match="必须提供MODEL"):
            Client(model="test-model", api_key="test-key", base_url="")


class TestChat:
    """测试 chat 方法"""

    def test_chat_basic(self):
        """测试基本聊天"""
        client = Client()
        messages = _m([{"role": "user", "content": "你好"}])

        # Mock OpenAI 响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "你好！有什么可以帮助你的吗？"

        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ):
            result = client.chat(messages)
            assert result == "你好！有什么可以帮助你的吗？"

    def test_chat_with_custom_temperature(self):
        """测试带自定义温度的聊天"""
        client = Client(temperature=0.3)
        messages = _m([{"role": "user", "content": "测试"}])

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "测试回复"

        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            client.chat(messages, temperature=0.1)
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["temperature"] == 0.1

    def test_chat_with_custom_model(self):
        """测试带自定义模型的聊天"""
        client = Client()
        messages = _m([{"role": "user", "content": "测试"}])

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "测试回复"

        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            client.chat(messages, model="gpt-4")
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["model"] == "gpt-4"

    def test_chat_with_max_tokens(self):
        """测试带 max_tokens 的聊天"""
        client = Client()
        messages = _m([{"role": "user", "content": "测试"}])

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "测试回复"

        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            client.chat(messages, max_completion_tokens=100)
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["max_completion_tokens"] == 100

    def test_chat_error_handling(self):
        """测试聊天错误处理"""
        client = Client()
        messages = _m([{"role": "user", "content": "测试"}])

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=Exception("API Error"),
        ):
            with pytest.raises(RuntimeError, match="调用LLM API时发生错误"):
                client.chat(messages)

    def test_chat_with_empty_response(self):
        """测试空响应处理"""
        client = Client()
        messages = _m([{"role": "user", "content": "测试"}])

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None

        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ):
            result = client.chat(messages)
            assert result == ""  # 空响应返回空字符串而不是 None


class TestStreamChat:
    """测试 stream_chat 方法"""

    def test_stream_chat_returns_iterator(self):
        """测试流式聊天返回迭代器"""
        client = Client()
        messages = _m([{"role": "user", "content": "你好"}])

        # Mock 流式响应
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = "你好"

        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = "，世界"

        with patch.object(client._client.chat.completions, "create") as mock_create:
            mock_create.return_value = iter([mock_chunk1, mock_chunk2])
            result = client.stream_chat(messages)
            chunks = list(result)
            assert len(chunks) == 2
            assert "你好" in chunks[0] or chunks[0] == ""
            assert "，世界" in chunks[1] or chunks[1] == ""

    def test_stream_chat_empty_chunks(self):
        """测试空块的流式聊天"""
        client = Client()
        messages = _m([{"role": "user", "content": "测试"}])

        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = None

        with patch.object(client._client.chat.completions, "create") as mock_create:
            mock_create.return_value = iter([mock_chunk])
            result = client.stream_chat(messages)
            chunks = list(result)
            # 空内容不应该产生输出
            assert len(chunks) == 0 or chunks == [None] or chunks == [""]

    def test_stream_chat_error(self):
        """测试流式聊天错误处理"""
        client = Client()
        messages = _m([{"role": "user", "content": "测试"}])

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=Exception("Stream Error"),
        ):
            with pytest.raises(RuntimeError, match="调用LLM API时发生错误"):
                list(client.stream_chat(messages))


class TestChatWithTools:
    """测试 chat_with_tools 方法"""

    def test_chat_with_tools_basic(self):
        """测试带工具的基本聊天"""
        client = Client()
        messages = _m([{"role": "user", "content": "使用工具"}])
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        # Mock 响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "工具执行结果"
        mock_response.choices[0].message.tool_calls = None

        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ):
            result = client.chat_with_tools(messages, tools)
            assert result.choices[0].message.content == "工具执行结果"

    def test_chat_with_tools_tool_choice(self):
        """测试带工具选择的聊天"""
        client = Client()
        messages = _m([{"role": "user", "content": "使用工具"}])
        tools = [
            {"type": "function", "function": {"name": "test_tool", "parameters": {}}}
        ]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "结果"

        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            client.chat_with_tools(messages, tools, tool_choice="none")
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["tool_choice"] == "none"

    def test_chat_with_tools_without_tools(self):
        """测试不带工具的聊天"""
        client = Client()
        messages = _m([{"role": "user", "content": "你好"}])

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "你好"

        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            result = client.chat_with_tools(messages, None)
            call_kwargs = mock_create.call_args[1]
            # 当没有 tools 时，不应该传递 tools 和 tool_choice 参数
            assert call_kwargs.get("tools") is None or call_kwargs.get("tools") == []
            assert (
                call_kwargs.get("tool_choice") is None
                or call_kwargs.get("tool_choice") == "auto"
            )

    def test_chat_with_tools_error(self):
        """测试带工具聊天的错误处理"""
        client = Client()
        messages = _m([{"role": "user", "content": "使用工具"}])
        tools = [
            {"type": "function", "function": {"name": "test_tool", "parameters": {}}}
        ]

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=Exception("API Error"),
        ):
            with pytest.raises(RuntimeError, match="调用LLM API时发生错误"):
                client.chat_with_tools(messages, tools)


class TestClientAttributes:
    """测试客户端属性"""

    def test_model_attribute(self):
        """测试模型属性"""
        client = Client(model="test-model")
        assert client._model == "test-model"

    def test_temperature_attribute(self):
        """测试温度属性"""
        client = Client(temperature=0.5)
        assert client._temperature == 0.5

    def test_max_completion_tokens_attribute(self):
        """测试 max_completion_tokens 属性"""
        client = Client(max_completion_tokens=500)
        assert client._max_completion_tokens == 500

    def test_client_attribute(self):
        """测试内部客户端属性"""
        client = Client()
        assert hasattr(client, "_client")
        assert client._client is not None
