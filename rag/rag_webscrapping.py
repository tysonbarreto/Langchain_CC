from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_community.document_loaders import WebBaseLoader, FireCrawlLoader

from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

llm= ChatOpenAI(model="gpt-4o-mini")

current_dir = os.path.dirname(Path(__file__))
presistent_directory=os.path.join(current_dir,"db/chroma_db_crawled")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



def create_vectore_store(url):
    api_key=os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")
    print("Beging crawing the website...")
    loader = FireCrawlLoader(
        api_key=api_key, url=url, mode="scrape"
    )
    docs = loader.load()
    print("Finished crawling the website.")
    
    for doc in docs:
        for k, v in doc.metadata.items():
            if isinstance(v, list):
                doc.metadata[k]=",".join(map(str,v))
                
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                             chunk_overlap=0)
    split_docs=text_splitter.split_documents(documents=docs) 
    
    
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(split_docs)}")
    print(f"Sample chunk:\n{split_docs[0].page_content}\n")

def query_vector_store(query):
    db = Chroma(persist_directory=presistent_directory,
            embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type='similarity',
        search_kwargs={"k":4}
    )
    
    relevant_docs = retriever.invoke(query)
    
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
            
if __name__=="__main__": 
    current_dir = os.path.dirname(Path(__file__))
    presistent_directory=os.path.join(current_dir,"db/chroma_db_crawled")
    url = "https://en.wikipedia.org/wiki/PlayStation"
    if not os.path.exists(presistent_directory):
        create_vectore_store(url)
    else:
        print(
            f"Vector store {presistent_directory} already exists. No need to initialize.")
    query = "What is the page about?"
    query_vector_store(query)

 