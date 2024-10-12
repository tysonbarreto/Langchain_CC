import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv
from pathlib import Path
from pydantic.dataclasses import dataclass
from typing import List, Optional, Union

load_dotenv()

#model = ChatOpenAI(model="gpt-4o-mini")


# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def query_vector_store(query,
                       store_name,
                       persist_directory,
                       embedding_function,
                       search_type,
                       search_kwargs
                       ):
    
    if os.path.exists(persist_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db=Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        retriever=db.as_retriever(
            search_kwargs=search_kwargs,
            search_type=search_type
        )
        relevant_docs=retriever.invoke(query)
        
        for i, doc in enumerate(relevant_docs,1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")

query = "How did Juliet die?"

"""
    # 1. Similarity Search
    # This method retrieves documents based on vector similarity.
    # It finds the most similar documents to the query vector based on cosine similarity.
    # Use this when you want to retrieve the top k most similar documents.
"""

# print("\n--- Using Similarity Score Threshold ---")
# query_vector_store(
#     store_name="chroma_db_with_metadata",
#     persist_directory=persistent_directory,
#     query=query,
#     embedding_function=embeddings,
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 3, "score_threshold": 0.1}
# )

# print("Querying demonstrations with different search types completed.")


"""
    # 2. Max Marginal Relevance (MMR)
    # This method balances between selecting documents that are relevant to the query and diverse among themselves.
    # 'fetch_k' specifies the number of documents to initially fetch based on similarity.
    # 'lambda_mult' controls the diversity of the results: 1 for minimum diversity, 0 for maximum.
    # Use this when you want to avoid redundancy and retrieve diverse yet relevant documents.
    # Note: Relevance measures how closely documents match the query.
    # Note: Diversity ensures that the retrieved documents are not too similar to each other,
    #       providing a broader range of information.
"""
print("\n--- Using Similarity Score Threshold ---")
query_vector_store(
    store_name="chroma_db_with_metadata",
    persist_directory=persistent_directory,
    query=query,
    embedding_function=embeddings,
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 4, "lambda_mult": 0.5}
)

print("Querying demonstrations with different search types completed.")