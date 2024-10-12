from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from textwrap import dedent

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")


# PART 1
# print("-------------Prompt from template-------------------")
# template="Tell me a joke about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({"topic":"cats"})

# result = model.invoke(prompt)

# print(result.content)

# PART 2
# print("-------------Prompt from template-------------------")
# template="""You are a helpful assistant.
#             Human: Tell me a {adjective} short story about a {animal}.
#             Assistant:"""

# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({"adjective":"funny","animal":"panda"})

# result = model.invoke(prompt)

# print(result.content)

# PART 3: Prompt with System and Human Messages (Using Tuples)
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = model.invoke(prompt)
print(result.content)
