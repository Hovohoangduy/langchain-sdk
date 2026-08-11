import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="./env/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7

class WeatherInput(BaseModel):
    """Input for weather queries"""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result

agent = create_agent(
    model="gpt-5.4-mini",
    tools=[get_weather],
)

config = {"configurable": {"thread_id": str(uuid7())}}


response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is weather in Tokyo?"}]},
)
print(response)



###
## Context
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.utils.uuid import uuid7
from langchain_openai import ChatOpenAI

USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com",
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com",
    },
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information"""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return (
            f"Account holder: {user['name']}\n"
            f"Type: {user['account_type']}\n"
            f"Balance: ${user['balance']}"
        )
    return "User not found"

model = ChatOpenAI(model="gpt-5.4-mini")
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my current balance?"}]},
    # config={"configurable": {"thread_id": str(uuid7())}},
    context=UserContext(user_id="user123"),
)
print(result)


###
## Long-term memory (Store)
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI

# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# update memory
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

model_ltm = ChatOpenAI(model="gpt-5.4-mini")
store = InMemoryStore()
agent_ltm = create_agent(
    model_ltm,
    tools=[get_user_info, save_user_info],
    store=store,
)

# first session: save user info
agent_ltm.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 23, email: foo@chain.dev"}]
})

# second session: get user info
agent_ltm.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})


###
## return direcly from a tool
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

@tool
def fetch_oder_status(order_id: str) -> str:
    """Fetch the current status of customer order."""
    return f"Order {order_id} is shipped and will arrive in 2 days."

agent = create_agent(
    ChatOpenAI(model="gpt-5.4-mini"),
    tools=[fetch_oder_status],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the status of order #12345?"}]
})
result


