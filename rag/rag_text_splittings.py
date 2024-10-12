import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.documents import Document

from dotenv import load_dotenv
from pathlib import Path

from pydantic.dataclasses import dataclass
from typing import List

load_dotenv()

#model = ChatOpenAI(model="gpt-4o-mini")

# Define the directory containing the text file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

print("Books directory: {books_dir}\n")
print("Persistent directory: {persistent_directory}\n")


# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the text content from the file
loader = TextLoader(file_path=file_path,encoding="utf-8")
documents = loader.load()

def create_vector_store(splitted_docs,store_name):
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
class VectoreStores:
    documents:List[Document]
    query:str
    
    @property
    def text_splitter(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents=self.documents)
        create_vector_store(docs,"chroma_db_char")
    
    @property
    def sent_splitter(self):
        sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = sent_splitter.split_documents(documents=self.documents)
        create_vector_store(docs,"chroma_db_sent")
    
    @property
    def recur_splitter(self):
        recur_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = recur_splitter.split_documents(documents=self.documents)
        create_vector_store(docs,"chroma_db_recur")
    
    @property
    def token_splitter(self):
        token_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = token_splitter.split_documents(documents=self.documents)
        create_vector_store(docs,"chroma_db_token")

    @staticmethod
    def query_vectore_store(query,store_name):
        persistent_directory = os.path.join(current_dir,f"db/{store_name}")
        if os.path.exists(persistent_directory):
            print(f"\n--- Querying the Vector Store {store_name} ---")
            
            db=Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        
            retriever = db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k":3,"score_threshold":0.2}
            )

            relevant_docs = retriever.invoke(query)

            print("\n------Relevant Documents--------\n")
            for i, doc in enumerate(relevant_docs,start=1):
                print(f"Document {i}:\n{doc.page_content}\n")
                if doc.metadata:
                    print(f"Source: {doc.metadata.get('source','Unknown')}\n")

    def query_all_stores(self):
        self.text_splitter
        self.sent_splitter
        self.recur_splitter
        self.token_splitter
        self.query_vectore_store(self.query, "chroma_db_char")
        self.query_vectore_store(self.query, "chroma_db_sent")
        self.query_vectore_store(self.query, "chroma_db_recur")
        self.query_vectore_store(self.query, "chroma_db_token")
    
vectorestores_=VectoreStores(documents=documents, query="How did Juliet die?")
        
vectorestores_.query_all_stores()


# print(f"\n{'#'*60}RAW OUTPUT{'#'*60}\n{documents}")