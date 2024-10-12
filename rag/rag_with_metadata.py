import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

#model = ChatOpenAI(model="gpt-4o-mini")

current_dir = Path(__file__).parent
books_dir = os.path.join(current_dir,"books")
persistent_directory = os.path.join(current_dir,"db/chroma_db_with_metadata")

embeddings=None

print("Books directory: {books_dir}\n")
print("Persistent directory: {persistent_directory}\n")

if not os.path.exists(persistent_directory):
    print("Persistent Directory does not exist, Intializing vector store...")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The file {books_dir} does not exist. Please check path..."
        )
    books_list = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    
    documents=[]
    for book_file in books_list:
        file_path = os.path.join(books_dir,book_file)
        loader = TextLoader(file_path=file_path,encoding='utf-8')
        book_docs=loader.load()
        for doc in book_docs:
            doc.metadata={"source":book_file}
            documents.append(doc)

    text_spplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_spplitter.split_documents(documents=documents)
    
    print("\n------- Document Chunk Information------\n")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0]}")
    
    print("\n------- Creating Embeddings------\n")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    print("\n------- Embeddings Created------\n")
    
    print("\n-------Creating VectorStore------\n")
    db = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persistent_directory
    )
    print("\n-------VectorStore created successfully------\n")
    
else:
    print("Vector store already exists. No need to initialize.")

if embeddings==None:
    print("\n------- Creating Embeddings------\n")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

query = "How did Juliet die?" 
    
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
    
# print(f"\n{'#'*60}RAW OUTPUT{'#'*60}\n{documents}")