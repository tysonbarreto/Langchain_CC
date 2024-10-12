from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# PART#1

# template = "Tell me a joke about {topic}"
# prompt_template = ChatPromptTemplate.from_template(template)

# print("------- Prompt from Template ---------")
# prompt = prompt_template.invoke({"topic":"cats"})
# print(prompt_template)
# print(f"{'#'*60}")
# print(prompt)

#PART 2
# This does work:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 3 jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers"})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)