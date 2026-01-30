import os
from typing import Optional, Dict, Any
from pydantic import BaseModel


class Config(BaseModel):
    # LLM配置
    model: str = os.getenv("MODEL", "")
    temperature: float = 0.7
    max_completion_tokens: Optional[int] = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # 其他配置
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        max_tokens_str = os.getenv("MAX_COMPLETION_TOKENS")
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_completion_tokens=int(max_tokens_str) if max_tokens_str else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()
