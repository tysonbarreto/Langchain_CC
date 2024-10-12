from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

llm= ChatOpenAI(model="gpt-4o-mini")

current_dir = os.path.dirname(Path(__file__).parent)
presistent_directory=os.path.join(current_dir,"db/chroma_db_with_metadata")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db=Chroma(persist_directory=presistent_directory,
          embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)
contextualize_q_system_prompt=(
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)

history_aware_prompt=create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

qa_system_prompt=(
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)


qa_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)

querstion_answer_chain=create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

rag_chain=create_retrieval_chain(retriever = history_aware_prompt,
                                 combine_docs_chain=querstion_answer_chain)

def continual_chain():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query=input("You: ")
        if query.lower()=="exit":
            break
        result=rag_chain.invoke({"input":query,"chat_history":chat_history})
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

if __name__=="__main__":
    continual_chain()
            