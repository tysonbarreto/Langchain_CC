from langchain import hub
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from dotenv import load_dotenv

load_dotenv()

# Functions for the tools
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"


def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]


def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    return a + b


class ContenteStringsArgs(BaseModel):
    a: str=Field(description="First String")
    b: str=Field(description="Second String")
    

tools = [
    Tool(
        name="GreetUser",  # Name of the tool
        func=greet_user,  # Function to execute
        description="Greets the user by name.",  # Description of the tool
    ),
    # Use Tool for another simple function with a single input parameter.
    Tool(
        name="ReverseString",  # Name of the tool
        func=reverse_string,  # Function to execute
        description="Reverses the given string.",  # Description of the tool
    ),
    StructuredTool.from_function(
        func=concatenate_strings,
        name="ContenteStrings",
        description="Concatenates two strings.",
        args_schema=ContenteStringsArgs
    )
    
]

llm=ChatOpenAI(model="gpt-4o-mini")

prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_erros=True
)

response = agent_executor.invoke({"input":"Greet Alice"})
print("Response for 'Greet Alice': ",response)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)