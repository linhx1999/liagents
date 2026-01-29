"""测试 Message 消息类"""

import pytest
from datetime import datetime
from typing import cast

from liagents.core.message import Message, MessageRole


def _r(role: str) -> MessageRole:
    """将字符串转换为 MessageRole 类型"""
    return cast(MessageRole, role)


class TestMessageRole:
    """测试消息角色类型"""

    def test_message_role_literal(self):
        """测试消息角色类型定义"""
        # MessageRole 是 Literal 类型，应该能用于类型注解
        from typing import Literal

        # 验证 Literal 类型的定义
        assert isinstance(MessageRole, type) or hasattr(MessageRole, "__origin__")

    def test_valid_roles(self):
        """测试有效的角色值"""
        # 可以使用这些字符串作为角色
        roles = ["user", "assistant", "system", "tool"]
        for role in roles:
            msg = Message(role=_r(role), content="测试")
            assert msg.role == role


class TestMessageInit:
    """测试 Message 初始化"""

    def test_init_with_required_fields(self):
        """测试使用必需字段初始化"""
        msg = Message(role=_r("user"), content="测试消息")
        assert msg.role == "user"
        assert msg.content == "测试消息"
        assert msg.metadata is not None
        assert "timestamp" in msg.metadata  # type: ignore[operator]

    def test_init_with_metadata(self):
        """测试带元数据的初始化"""
        custom_metadata = {"source": "test", "priority": 1}
        msg = Message(role=_r("assistant"), content="回复", metadata=custom_metadata)
        assert msg.metadata == custom_metadata
        assert "timestamp" not in msg.metadata  # type: ignore[operator]

    def test_init_with_custom_timestamp(self):
        """测试带自定义时间戳的初始化"""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        msg = Message(role=_r("user"), content="测试", timestamp=custom_time)
        assert msg.metadata["timestamp"] == custom_time  # type: ignore[index]

    def test_init_with_all_kwargs(self):
        """测试使用所有关键字参数初始化"""
        custom_time = datetime(2024, 1, 1)
        msg = Message(
            role=_r("user"),
            content="测试",
            metadata={"key": "value"},
            timestamp=custom_time,
        )
        assert msg.role == "user"
        assert msg.content == "测试"
        # metadata 只包含传入的值，不包含 timestamp
        assert msg.metadata == {"key": "value"}


class TestMessageToDict:
    """测试消息转换为字典"""

    def test_to_dict_basic(self):
        """测试基本转换"""
        msg = Message(role=_r("user"), content="测试消息")
        result = msg.to_dict()
        assert result == {"role": "user", "content": "测试消息"}

    def test_to_dict_excludes_metadata(self):
        """测试转换后不包含 metadata"""
        msg = Message(role=_r("assistant"), content="回复", metadata={"extra": "data"})
        result = msg.to_dict()
        assert "metadata" not in result
        assert "extra" not in result

    def test_to_dict_all_roles(self):
        """测试所有角色的转换"""
        for role in ["user", "assistant", "system", "tool"]:
            msg = Message(role=_r(role), content="测试")
            result = msg.to_dict()
            assert result["role"] == role
            assert result["content"] == "测试"


class TestMessageStr:
    """测试消息字符串表示"""

    def test_str_representation(self):
        """测试字符串表示格式"""
        msg = Message(role=_r("user"), content="你好")
        result = str(msg)
        assert "[user]" in result
        assert "你好" in result

    def test_str_all_roles(self):
        """测试所有角色的字符串表示"""
        for role in ["user", "assistant", "system", "tool"]:
            msg = Message(role=_r(role), content="测试")
            result = str(msg)
            assert f"[{role}]" in result


class TestMessageEquality:
    """测试消息相等性"""

    def test_equal_messages(self):
        """测试相等的消息"""
        msg1 = Message(role=_r("user"), content="测试")
        msg2 = Message(role=_r("user"), content="测试")
        # 内容相同但 metadata 中的时间戳可能不同
        assert msg1.role == msg2.role
        assert msg1.content == msg2.content

    def test_different_roles(self):
        """测试不同角色的消息"""
        msg1 = Message(role=_r("user"), content="测试")
        msg2 = Message(role=_r("assistant"), content="测试")
        assert msg1.role != msg2.role

    def test_different_content(self):
        """测试不同内容的消息"""
        msg1 = Message(role=_r("user"), content="消息1")
        msg2 = Message(role=_r("user"), content="消息2")
        assert msg1.content != msg2.content


class TestMessageEdgeCases:
    """测试消息边界情况"""

    def test_empty_content(self):
        """测试空内容"""
        msg = Message(role=_r("user"), content="")
        assert msg.content == ""
        result = msg.to_dict()
        assert result["content"] == ""

    def test_long_content(self):
        """测试长内容"""
        long_content = "a" * 10000
        msg = Message(role=_r("user"), content=long_content)
        assert len(msg.content) == 10000
        assert msg.to_dict()["content"] == long_content

    def test_special_characters_in_content(self):
        """测试内容中的特殊字符"""
        special_content = "你好世界\n\t!@#$%^&*()"
        msg = Message(role=_r("user"), content=special_content)
        assert msg.content == special_content

    def test_unicode_content(self):
        """测试 Unicode 内容"""
        unicode_content = "中文测试 🎉 αβγ 🚀"
        msg = Message(role=_r("user"), content=unicode_content)
        assert msg.content == unicode_content
