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

class MultiEmbeddings:
    
    @property
    def openaiembeddings(self):
        return OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )
    
    @property
    def hfmbeddings(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

# Define the directory containing the text file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

openaiembeddings= MultiEmbeddings().openaiembeddings
hfembeddings= MultiEmbeddings().hfmbeddings


print(f"Books directory: {file_path}\n")
print(f"Persistent directory: {db_dir}\n")


# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the text content from the file
loader = TextLoader(file_path=file_path,encoding="utf-8")
documents = loader.load()

def create_vector_store(splitted_docs,store_name,embeddings):
    persistent_directory = os.path.join(current_dir,f"db/{store_name}")
    if not os.path.exists(persistent_directory):
        print("Persistent Directory does not exist, Intializing vector store...")
        print("-------Creating VectorStore------")
        db = Chroma.from_documents(
            documents=splitted_docs, embedding=embeddings, persist_directory=persistent_directory
        )
        db=None
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(f"Vector store {store_name} already exists. No need to initialize.")

@dataclass
class MultiEmbedStores:
    documents:List[Document]
    query:str
    
    @property
    def recur_splitter(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = splitter.split_documents(documents=self.documents)
        create_vector_store(docs,"chroma_db_openaiembed",openaiembeddings)
        create_vector_store(docs,"chroma_db_hfembed",hfembeddings)
        
    
    @staticmethod
    def query_vectore_store(query,store_name,embeddings):
        persistent_directory = os.path.join(current_dir,f"db/{store_name}")
        if os.path.exists(persistent_directory):
            print(f"\n--- Querying the Vector Store {store_name} ---")
            
            db=Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        
            retriever = db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k":5,"score_threshold":0.1}
            )

            relevant_docs = retriever.invoke(query)

            print("\n------Relevant Documents--------\n")
            for i, doc in enumerate(relevant_docs,start=1):
                print(f"Document {i}:\n{doc.page_content}\n")
                if doc.metadata:
                    print(f"Source: {doc.metadata.get('source','Unknown')}\n")
                    
    def query_all_embeddings(self):
        self.recur_splitter
        self.query_vectore_store(self.query, "chroma_db_openaiembed",openaiembeddings)
        self.query_vectore_store(self.query, "chroma_db_hfembed",hfembeddings)
    
multiembeds=MultiEmbedStores(documents=documents, query="How did Juliet die?")
        
multiembeds.query_all_embeddings()

