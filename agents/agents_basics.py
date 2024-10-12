from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
import datetime

load_dotenv()


def get_current_time(*args,**kwargs):
    return datetime.datetime.now().strftime("%I:%M %p")


if __name__=="__main__":
    
    tools=[
        Tool(
            name="Time",
            func=get_current_time,
            description="Useful for when you need to know the current time."
        )
    ]
    
    prompt = hub.pull("hwchase17/react")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )
    
    response = agent_executor.invoke({"input":"What is the time?"})
    
    print("AI reponse: ",response['output'])