import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="./env/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

def get_user_info() -> str:
    """Look up information about the current user."""
    return "No user profile on file."

agent = create_agent(
    model="gpt-5.4-mini",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),
)

thread_config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    thread_config
)["messages"][-1].content

print(response)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?."}]},
    thread_config
)["messages"][-1].content

print(response)


###
## Customizing agent memory
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5.4-mini",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

response = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme", "dark"}
    },
    {"configurable": {"thread_id": "1"}}
)
print(response["messages"][-1].content)