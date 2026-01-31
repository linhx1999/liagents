# liagents

基于 Python 3.12+ 的 AI Agents 框架，支持强化学习训练。

## 特性

- 🤖 多范式 Agent 支持（ReAct、反思、规划等）
- 🧪 RL 训练与评估（支持 LoRA）
- 🔧 可扩展的工具系统
- ⚙️ 统一的 LLM 接口

## 快速开始

### 安装依赖

```bash
# 安装基础依赖
uv sync

# 安装开发依赖（包含 black、pytest 等）
uv sync --group dev

# 安装可选依赖
uv sync --all-groups

# 单独安装特定可选依赖组
uv pip install -e ".[rl]"      # RL 训练
uv pip install -e ".[example]" # 示例代码依赖
```

### 配置环境变量

```bash
export LLM_MODEL_ID="your-model-id"
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
```

### 基础使用

```python
from liagents.core.client import Client
from liagents.agents.simple_agent import SimpleAgent

client = Client(model="gpt-4", api_key="your-key")
agent = SimpleAgent(name="assistant", client=client)
response = agent.run("你好！")
```

## RL 训练

### 训练模型

```python
from liagents.rl import RLTrainer

trainer = RLTrainer("/path/to/model")
trainer.load_dataset("./examples/datasets/gsm8k")
trainer.train(algorithm="sft", epochs=3)
```

### 评估模型

```python
# 评估训练后的模型
result = trainer.evaluate(max_samples=100)
```

### 查看 TensorBoard 日志

训练过程中会自动生成 TensorBoard 日志，查看训练曲线：

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/Qwen3-0.6B/20260131-082530/runs

# 然后访问 http://localhost:6006
```

## 项目结构

```
src/liagents/
├── core/       # 核心基础设施（Agent、Client、配置等）
├── tools/      # 工具系统
├── agents/     # 预置 Agent 实现
└── rl/         # RL 训练模块
```

## 开发

```bash
# 代码格式化
black src/

# 运行测试
pytest
```

## 许可证

LICENSE
