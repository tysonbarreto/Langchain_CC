from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
import datetime
from wikipedia import summary

load_dotenv()


def get_current_time(*args, **kwargs):
    return datetime.datetime.now().strftime("%I:%M %p")

def search_wikipedia(query):
    try:
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that"

if __name__=="__main__":
    llm = ChatOpenAI(name="gpt-4o-mini")
    tools=[
        Tool(
            name="Time",
            func=get_current_time,
            description="Useful for when you need to know the current time."
        ),
        Tool(
            name="Wikipedia",
            func=search_wikipedia,
            description="Useful for when you need to know information about a topic."
        )
    ]
    prompt = hub.pull("hwchase17/structured-chat-agent")
    
    memory = ConversationBufferMemory(llm=llm, tools=tools, prompt=prompt)
    
    agent = create_structured_chat_agent(
        llm=llm,
        tools=tools, 
        prompt=prompt
    )
    agent_executor=AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory, #use to maintain context in memory
        handle_parsing_errors=True, #handle any parsing errors gracefully
    )
    
    initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
    memory.chat_memory.add_message(SystemMessage(content=initial_message))

    # Chat Loop to interact with the user
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # Add the user's message to the conversation memory
        memory.chat_memory.add_message(HumanMessage(content=user_input))

        # Invoke the agent with the user input and the current chat history
        response = agent_executor.invoke({"input": user_input})
        print("Bot:", response["output"])

        # Add the agent's response to the conversation memory
        memory.chat_memory.add_message(AIMessage(content=response["output"]))