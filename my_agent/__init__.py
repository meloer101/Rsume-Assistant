# 延迟导入 agent：只有访问 root_agent 时才加载，避免 run ingest 时触发 LLM 环境变量校验。

def __getattr__(name: str):
    if name == "root_agent":
        from .agent import root_agent
        return root_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
