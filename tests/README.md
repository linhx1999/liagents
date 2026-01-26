# 测试说明

## 安装测试依赖

使用 `uv` 安装开发依赖（包含测试工具）：

```bash
uv pip install -e ".[dev]"
```

或者单独安装测试依赖：

```bash
uv pip install pytest pytest-mock
```

## 运行测试

运行所有测试：

```bash
pytest
```

运行特定测试文件：

```bash
pytest tests/test_simple_agent.py
```

运行特定测试类：

```bash
pytest tests/test_simple_agent.py::TestSimpleAgentInit
```

运行特定测试方法：

```bash
pytest tests/test_simple_agent.py::TestSimpleAgentInit::test_init_basic
```

查看测试覆盖率（需要安装 pytest-cov）：

```bash
pytest --cov=liagents --cov-report=html
```

## 测试结构

```
tests/
├── __init__.py                 # 测试包初始化
├── agents/                     # Agent 测试目录
│   └── __init__.py
├── test_simple_agent.py       # SimpleAgent 测试文件
└── README.md                   # 本文件
```

## SimpleAgent 测试覆盖

`test_simple_agent.py` 包含以下测试类别：

### 1. 初始化测试 (`TestSimpleAgentInit`)
- 基本初始化
- 不提供系统提示词
- 带工具注册表初始化
- 默认创建工具注册表

### 2. 系统提示词测试 (`TestGetEnhancedSystemPrompt`)
- 不带工具时的提示词
- 无提示词和工具时的默认提示词
- 带工具时的提示词增强
- 空工具注册表的处理

### 3. 工具调用解析测试 (`TestParseToolCalls`)
- 解析单个工具调用
- 解析多个工具调用
- 没有工具调用的情况
- 工具调用在文本中间的情况

### 4. 工具执行测试 (`TestExecuteToolCall`)
- 成功执行工具
- 工具不存在的情况
- 没有工具注册表的情况
- 工具执行异常的处理

### 5. 运行方法测试 (`TestRun`)
- 不带工具的运行
- 带工具调用的运行
- 达到最大迭代次数
- 带历史记录的运行
- 传递额外参数

### 6. 工具管理测试 (`TestToolManagement`)
- 添加工具
- 移除工具
- 列出工具

### 7. 流式运行测试 (`TestStreamRun`)
- 流式返回片段
- 保存到历史记录

### 8. 继承方法测试 (`TestInheritedMethods`)
- 添加消息
- 清空历史记录
- 获取历史记录
- 字符串表示

### 9. 边界情况测试 (`TestEdgeCases`)
- 空输入
- 格式错误的工具调用
- 参数中的特殊字符

## 编写新测试

在编写新测试时，请遵循以下约定：

1. 使用 `pytest` 作为测试框架
2. 使用 `unittest.mock` 进行 mock
3. 测试类名以 `Test` 开头
4. 测试方法名以 `test_` 开头
5. 为测试添加清晰的文档字符串
6. 使用 fixtures 提供可复用的测试数据

## Mock 对象说明

测试中使用了以下 mock 对象：

- `mock_client`: 模拟 LLM 客户端
- `mock_config`: 模拟配置对象
- `mock_tool`: 模拟工具对象
- `mock_tool_registry`: 模拟工具注册表
- `simple_agent`: 不带工具的 SimpleAgent 实例
- `simple_agent_with_tools`: 带工具的 SimpleAgent 实例
