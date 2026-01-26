# liagents

## 添加依赖

### 安装项目依赖

本项目使用 `pip` 进行依赖管理。

#### 运行时依赖

运行时依赖已经包含在 `pyproject.toml` 中，主要包括：

- `openai<2.0.0` - OpenAI API 客户端

安装运行时依赖：

```bash
pip install .
```

#### 开发依赖

开发时需要额外的工具进行代码格式化。主要依赖：

- `black` - 代码格式化工具

安装开发依赖：

```bash
pip install -e ".[dev]"
```

或者使用开发模式安装：

```bash
pip install -e .
```

### 环境要求

- Python >= 3.12
