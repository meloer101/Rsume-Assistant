from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
import os
from .prompt import Writing_agent_prompt
from .rag_tool import retrieve_pm_knowledge  # 加这一行

test_model = LiteLlm(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_MODEL_API_KEY"),
    api_base=os.getenv('LLM_BASE_URL', {}),
)

writing_agent = LlmAgent(
    model=test_model,
    name='writing_agent',
    description='A helpful assistant for user questions.',
    instruction=Writing_agent_prompt,
    tools=[retrieve_pm_knowledge]  # 加这一行
)

root_agent = writing_agent


