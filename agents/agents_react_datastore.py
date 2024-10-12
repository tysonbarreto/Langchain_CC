import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_retrieval_chain, create_history_aware_retriever, retrieval, history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from pathlib import Path

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
embeddings= OpenAIEmbeddings(model="text-embedding-3-small")

parent_dir = Path(__file__).parent.parent
persistent_directory= os.path.join(parent_dir,"rag","db","chroma_db_with_metadata")

if os.path.exists(persistent_directory):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embeddings)
else:
    raise FileNotFoundError(
        f"The directory {persistent_directory} does not exist. Please check the path."
    )


retriever = db.as_retriever(
    search_type='similarity', search_kwargs={"k":3}
)

contentualize_q_system_prompt=(
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contentualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",contentualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm= llm,
    retriever= retriever,
    prompt= contentualize_q_prompt,
)

qa_system_prompt=(
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Do not find answers outside the retrieved context."
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

qa_prompt =ChatPromptTemplate.from_messages(
    [
        ("system",qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm=llm,prompt=qa_prompt
)

rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain
)

react_docstore_prompt = hub.pull("hwchase17/react")

tools=[
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input":input, "chat_history":kwargs.get("chat_history",[])}
        ),
        description="useful for when you need to answer questions about the context",
        
    )
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt
)

# memory = ConversationBufferMemory(llm=llm, prompt=react_docstore_prompt, tools=tools)
# memory.chat_memory.add_message(
#     SystemMessage(
#         content="You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
# ))


agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools,
    #memory=memory,
    handle_parsing_erros=True,
    verbose=True
)

chat_history=[]
while True:
    query=input("You: ")
    if query.lower()=="exit":
        break
    response = agent_executor.invoke({"input":query, "chat_history":chat_history})
    print(f"AI message: {response['output']}")
    
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))
    
    # memory.chat_memory.add_message(HumanMessage(content=query))
    # memory.chat_memory.add_message(AIMessage(content=response["output"]))

print(chat_history)
