from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")



message = [
    SystemMessage(content="Solve the following Math problems"),
    HumanMessage(content="What is 81 divided by 9?")
]
result = model.invoke(message)


print(result.content)
