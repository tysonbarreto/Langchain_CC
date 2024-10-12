from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel
from langchain_openai import ChatOpenAI
from enum import Enum
from pydantic.dataclasses import dataclass
from pydantic import Field, BaseModel


load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert product reviewer."),
        ("human","List the main features of the product {product_name}"),
    ]
)

@dataclass(frozen=True)
class feedback(str, Enum):
    pros: str = Field(default="pros")
    cons: str = Field(default="cons")

@dataclass
class Analyze:
    features: str
    # pros: feedback = feedback.pros
    # cons: feedback = feedback.cons
    
    def __post_init__(self):
        self.pros_template = ChatPromptTemplate.from_messages(
                                                    [
                                                        ('system','You are an expert product reviewer.'),
                                                        ('human','Given these features: {features}, list the pros of these features.'),
                                                    ]
                                                )
        self.cons_template = ChatPromptTemplate.from_messages(
                                                    [
                                                        ('system','You are an expert product reviewer.'),
                                                        ('human','Given these features: {features}, list the cons of these features.'),
                                                    ]
                                                )

    @property
    def pros(self):
        return self.pros_template.format_prompt(features = self.features)
    
    @property
    def cons(self):
        return self.cons_template.format_prompt(features = self.features)

    @staticmethod
    def combine_pros_cons(pros,cons):
        return f"Pros:\n{pros}\n\nCons:\n{cons}"
    
pros_branch_chain= (
    RunnableLambda(lambda x: Analyze(features=x).pros) | model | StrOutputParser()
)

cons_branch_chain= (
    RunnableLambda(lambda x: Analyze(features=x).cons) | model | StrOutputParser()
)

chain =(
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros":pros_branch_chain, "cons":cons_branch_chain})
    | RunnableLambda(lambda x: Analyze.combine_pros_cons(x["branches"]["pros"],x["branches"]["cons"]))
)

result = chain.invoke({"product_name":"MacBook Pro"})
# #Runnable is a task
# format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
# invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
# parse_output = RunnableLambda(lambda x: x.content)

# chain = RunnableSequence(first=format_prompt,middle=[invoke_model],last=parse_output)
# #chain = prompt_template | model | StrOutputParser()

# result = chain.invoke({"product_name":"Macbook pro"})

print(result)
