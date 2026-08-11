import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="./env/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


###
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("gpt-5-nano")

system_msg = SystemMessage("You are a helpful assistant")
human_msg = HumanMessage("Hello, how are you?")

messages = [system_msg, human_msg]
respose = model.invoke(messages)
respose


###
# Text prompts
respose = model.invoke("Write a haiku about string")
respose


###
# Messgaes prompts
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom...")
]

respose = model.invoke(messages)
respose


###
# Dictionary format
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
    {"role": "system", "content": "Cherry blossoms bloom..."}
]
respose = model.invoke(messages)
respose


###
# System message
system_msg = SystemMessage("You are helpful coding assistant.")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API")
]
respose = model.invoke(messages)
respose

# Context details
from langchain.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("""
You are seniot Python developer with expertise in web frameworks.
Always provide code examples and expalin your reasoning.
Be consice but thorough in your explainations.
""")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API")
]
respose = model.invoke(messages)
respose


###
# Human message
respose = model.invoke([
    HumanMessage("What is machine learning?")
])
respose


###
# AI Messages
from langchain.messages import AIMessage, SystemMessage, HumanMessage

# AI message manually
ai_msg = AIMessage("I'd be happy to help you with that question!")

# add conversation history
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg,
    HumanMessage("Great! What's 2+2?")
]
respose = model.invoke(messages)
respose


###
# Tool calls
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")
def get_weather(location: str) -> str:
    """Get weather at a location"""
    ...

model_with_tools = model.bind_tools([get_weather])
respose = model_with_tools.invoke("What's the weather in Paris?")

for tool_call in respose.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")



###
# Tool message
from langchain.messages import AIMessage
from langchain.messages import ToolMessage

ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

weather_result = "Sunny, 72°F"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"
)

messages = [
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,
    tool_message,
]

respose = model.invoke(messages)
respose


###
## Message content
from langchain.messages import HumanMessage

## String content
# human_message = HumanMessage("What is this?")

## Provider native-format
human_message = HumanMessage(content=[
    {"type": "text", "text": "What is this?"},
    {"type": "image_url", "image_url": {"url": "https://cdn.growthjockey.com/blogs/single-agent-system.png"}}
])

respose = model.invoke([human_message])
respose


###
## Standard content blocks
from langchain.messages import AIMessage

message = AIMessage(
    content=[
        {
            "type": "reasoning",
            "id": "rs_abc123",
            "summary": [
                {"type": "summary_text", "text": "summary 1"},
                {"type": "summary_text", "text": "summary 2"}
            ]
        },
        {"type": "text", "text": "...", "id": "msg_abc123"},
    ],
    respose_metadata={"model_provider": "openai"}
)
message.content_blocks


###
## Multi modal
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "url": "https://images.viblo.asia/96522d2b-b7ef-46f8-b66c-bdf9c6ba6e52.png"}
    ]
}

respose = model.invoke([message])
respose