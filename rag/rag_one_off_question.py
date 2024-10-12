import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
from pathlib import Path
from pydantic.dataclasses import dataclass
from typing import List, Optional, Union

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db=Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":2}
)

query = "What is the Romeo and Juliet story about? "

relevant_docs = retriever.invoke(query)

one_off_input=(
    "Here are some documents that might help answer the question:"
    + query
    + "\n\nRelevant Documents\n\n:"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\n Please provide an answer based only based on the provided docyments.\
    If the answer is not found in the documents, respond with 'I'm not sure'."
)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=one_off_input)
    
]

print("\n--- Generated Response ---")

result = model.invoke(messages)

print(result.content)
