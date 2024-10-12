from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a comedian who tell jokes about {topic}."),
        ("human","Tell me {joke_count} jokes."),
    ]
)

#Runnable is a task
uppercast_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")


chain = prompt_template | model | StrOutputParser() | uppercast_output | count_words
#chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"topic": "lawyers", "joke_count":3})

print(result)
