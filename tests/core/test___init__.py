"""测试 liagents 包初始化"""

import pytest


class TestPackageInit:
    """测试包初始化"""

    def test_hello_function_exists(self):
        """测试 hello 函数存在"""
        from liagents import hello

        assert callable(hello)

    def test_hello_returns_greeting(self):
        """测试 hello 函数返回正确的问候语"""
        from liagents import hello

        result = hello()
        assert result == "Hello from liagents!"
