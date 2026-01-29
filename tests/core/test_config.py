"""测试 Config 配置类"""

import os
import pytest

# 设置测试所需的环境变量
os.environ["MODEL"] = "test-model"

from liagents.core.config import Config


class TestConfigInit:
    """测试 Config 初始化"""

    def test_default_values(self):
        """测试默认值（model 从环境变量读取）"""
        config = Config()

        # LLM 配置默认值 - model 从环境变量读取
        assert config.model == "test-model"
        assert config.temperature == 0.7
        assert config.max_completion_tokens is None

        # 系统配置默认值
        assert config.debug is False
        assert config.log_level == "INFO"

        # 其他配置默认值
        assert config.max_history_length == 100

    def test_custom_values(self):
        """测试自定义值"""
        config = Config(
            model="gpt-4",
            temperature=0.5,
            max_completion_tokens=1000,
            debug=True,
            log_level="DEBUG",
            max_history_length=50,
        )

        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_completion_tokens == 1000
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.max_history_length == 50


class TestConfigFromEnv:
    """测试从环境变量创建配置"""

    def test_from_env_defaults(self):
        """测试默认环境变量值"""
        # 确保环境变量未设置
        os.environ.pop("DEBUG", None)
        os.environ.pop("LOG_LEVEL", None)
        os.environ.pop("TEMPERATURE", None)
        os.environ.pop("MAX_COMPLETION_TOKENS", None)

        config = Config.from_env()

        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.temperature == 0.7
        assert config.max_completion_tokens is None

    def test_from_env_with_values(self):
        """测试带值的从环境变量创建"""
        os.environ["DEBUG"] = "true"
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["TEMPERATURE"] = "0.3"
        os.environ["MAX_COMPLETION_TOKENS"] = "2000"

        config = Config.from_env()

        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.temperature == 0.3
        assert config.max_completion_tokens == 2000

        # 清理
        os.environ.pop("DEBUG", None)
        os.environ.pop("LOG_LEVEL", None)
        os.environ.pop("TEMPERATURE", None)
        os.environ.pop("MAX_COMPLETION_TOKENS", None)

    def test_from_env_invalid_temperature(self):
        """测试无效的温度值（当前实现会抛出异常）"""
        os.environ["TEMPERATURE"] = "invalid"

        # 当前实现会直接抛出异常，不会使用默认值
        with pytest.raises(ValueError):
            Config.from_env()

        os.environ.pop("TEMPERATURE", None)

    def test_from_env_invalid_max_tokens(self):
        """测试无效的最大 token 值（当前实现会抛出异常）"""
        os.environ["MAX_COMPLETION_TOKENS"] = "not_a_number"

        # 当前实现会直接抛出异常
        with pytest.raises(ValueError):
            Config.from_env()

        os.environ.pop("MAX_COMPLETION_TOKENS", None)


class TestConfigToDict:
    """测试配置转换为字典"""

    def test_to_dict_basic(self):
        """测试基本转换"""
        config = Config(model="test-model", temperature=0.5)
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["model"] == "test-model"
        assert result["temperature"] == 0.5

    def test_to_dict_all_fields(self):
        """测试所有字段的转换"""
        config = Config(
            model="gpt-4",
            temperature=0.7,
            max_completion_tokens=1000,
            debug=True,
            log_level="DEBUG",
            max_history_length=50,
        )
        result = config.to_dict()

        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.7
        assert result["max_completion_tokens"] == 1000
        assert result["debug"] is True
        assert result["log_level"] == "DEBUG"
        assert result["max_history_length"] == 50

    def test_to_dict_default_values(self):
        """测试默认值的转换"""
        # MODEL 环境变量已设置为 test-model
        config = Config()
        result = config.to_dict()

        # model 从环境变量读取
        assert result["model"] == "test-model"
        assert result["temperature"] == 0.7
        assert result["max_completion_tokens"] is None
        assert result["debug"] is False
        assert result["log_level"] == "INFO"
        assert result["max_history_length"] == 100


class TestConfigEquality:
    """测试配置相等性"""

    def test_equal_configs(self):
        """测试相等的配置"""
        config1 = Config(model="test", temperature=0.5)
        config2 = Config(model="test", temperature=0.5)
        assert config1.model == config2.model
        assert config1.temperature == config2.temperature

    def test_different_configs(self):
        """测试不同的配置"""
        config1 = Config(model="model1")
        config2 = Config(model="model2")
        assert config1.model != config2.model


class TestConfigEdgeCases:
    """测试配置边界情况"""

    def test_temperature_range(self):
        """测试温度范围"""
        # 有效温度
        config = Config(temperature=0.0)
        assert config.temperature == 0.0

        config = Config(temperature=1.0)
        assert config.temperature == 1.0

        config = Config(temperature=0.999)
        assert config.temperature == 0.999

    def test_model_from_env(self):
        """测试从环境变量读取模型"""
        # MODEL 环境变量已设置为 test-model
        config = Config()
        assert config.model == "test-model"

    def test_log_level_values(self):
        """测试日志级别值"""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = Config(log_level=level)
            assert config.log_level == level

    def test_max_history_length_boundary(self):
        """测试最大历史长度边界"""
        config = Config(max_history_length=0)
        assert config.max_history_length == 0

        config = Config(max_history_length=10000)
        assert config.max_history_length == 10000
