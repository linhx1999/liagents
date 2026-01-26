# liagents

一个基于 Python 3.12+ 的 AI Agents 框架，用于构建和执行智能代理。项目使用 OpenAI API 作为大语言模型后端。

## 特性

- 🤖 **多范式 Agent 支持** - 支持简单对话、ReAct、反思、规划等多种 Agent 模式
- 🔧 **灵活的工具系统** - 可扩展的工具注册和执行机制
- 📝 **消息历史管理** - 自动维护对话历史
- ⚙️ **统一的 LLM 接口** - 简化大语言模型调用
- 🧪 **完整的测试覆盖** - 单元测试确保代码质量

## 环境要求

- Python >= 3.12
- uv（推荐的包管理器）

## 快速开始

### 1. 安装依赖

本项目使用 `uv` 进行依赖管理。

#### 安装 uv

如果尚未安装 uv：

```bash
# macOS / Linux / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 安装项目依赖

```bash
# 安装运行时依赖
uv pip install .

# 安装开发依赖（包含测试和代码格式化工具）
uv pip install -e ".[dev]"
```

或者使用 uv 的同步功能（推荐）：

```bash
uv sync
```

### 2. 配置环境变量

在使用 LLM 功能前，需要设置以下环境变量：

```bash
export LLM_MODEL_ID="your-model-id"
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_TIMEOUT="60"  # 可选，默认 60 秒
```

建议使用 `.env` 文件或类似工具（如 `python-dotenv`）管理环境变量。

### 3. 使用示例

```python
from liagents.core.client import Client
from liagents.agents.simple_agent import SimpleAgent

# 创建客户端
client = Client(
    model="gpt-4",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"
)

# 创建 Agent
agent = SimpleAgent(
    name="assistant",
    client=client,
    system_prompt="你是一个有用的AI助手。"
)

# 运行对话
response = agent.run("你好！")
print(response)
```

## 项目结构

```
liagents/
├── src/liagents/
│   ├── core/              # 核心基础设施
│   │   ├── agent.py       # Agent 基类
│   │   ├── client.py      # LLM 客户端封装
│   │   ├── config.py      # 配置管理
│   │   ├── message.py     # 消息协议定义
│   │   └── exceptions.py  # 异常定义
│   ├── tools/             # 工具系统
│   │   ├── base.py        # 工具基类
│   │   ├── registry.py    # 工具注册中心
│   │   ├── chain.py       # 工具链执行
│   │   ├── async_executor.py  # 异步执行器
│   │   └── builtin/       # 内置工具
│   └── agents/            # 预置 Agent 实现
│       ├── simple_agent.py      # 简单对话 Agent
│       ├── react_agent.py       # ReAct 框架 Agent
│       ├── reflection_agent.py  # 反思型 Agent
│       └── plan_solve_agent.py  # 规划求解 Agent
├── tests/                 # 测试文件
├── examples/              # 示例代码
├── pyproject.toml         # 项目配置
├── uv.lock               # 依赖锁定文件
└── README.md             # 本文件
```

## 开发指南

### 代码格式化

项目使用 `black` 进行代码格式化：

```bash
# 格式化所有代码
black src/ tests/

# 检查格式是否符合规范
black --check src/ tests/
```

### 运行测试

使用 pytest 运行测试：

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_simple_agent.py

# 查看详细输出
pytest -v

# 查看测试覆盖率
pytest --cov=liagents --cov-report=html
```

详细的测试说明请参考 [tests/README.md](tests/README.md)。

### 添加新依赖

使用 uv 添加新依赖：

```bash
# 添加运行时依赖
uv pip add package-name

# 添加开发依赖
uv pip add --dev package-name

# 同步依赖（更新 uv.lock）
uv sync
```

### 创建自定义 Agent

所有 Agent 都应继承自 `Agent` 基类：

```python
from liagents.core.agent import Agent
from liagents.core.client import Client

class MyCustomAgent(Agent):
    def run(self, input_text: str, **kwargs) -> str:
        # 实现你的 Agent 逻辑
        return "响应内容"
```

### 创建自定义工具

工具需要继承 `Tool` 基类：

```python
from liagents.tools.base import Tool, ToolParameter

class MyTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="我的工具描述"
        )

    def run(self, parameters: dict) -> str:
        # 实现工具逻辑
        return "执行结果"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="param1",
                type="string",
                description="参数描述",
                required=True
            )
        ]
```

## 设计理念

- **模块化** - 每个组件职责清晰，易于扩展和替换
- **可组合** - 工具系统支持链式调用和组合
- **多范式** - 支持不同的 Agent 推理模式（ReAct、反思、规划等）
- **类型安全** - 使用类型注解提高代码可维护性

## 常见问题

### Q: 为什么使用 uv 而不是 pip？

A: uv 是一个更快的 Python 包管理器，由 Rust 编写，比 pip 快 10-100 倍。它还提供了更好的依赖解析和锁定机制。

### Q: 如何切换不同的 LLM 提供商？

A: 只需修改 `Client` 初始化时的 `base_url` 和 `api_key` 参数即可。任何兼容 OpenAI API 格式的提供商都可以使用。

### Q: 工具调用支持哪些格式？

A: SimpleAgent 支持类 OpenAI Function Calling 的 XML 标签格式，也支持原生 OpenAI Function Calling（在相应的 Agent 实现中）。

## 许可证

本项目采用 LICENSE 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

- 作者: linhx
- 邮箱: linhx1999@163.com
