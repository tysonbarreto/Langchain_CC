from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

print("Initializing Firestore Client....")

cred = credentials.Certificate('firebase.json')

app = firebase_admin.initialize_app(cred)

client = firestore.client()

load_dotenv()

PROJECT_ID="langchaincc"
SESSION_ID="user_session"
COLLECTION_NAME="chat_history"


print("Initializing Firestore Chat Message History....")
chat_history=FirestoreChatMessageHistory(
    session_id = SESSION_ID,
    collection = COLLECTION_NAME,
    client = client
)
print("Chat History Intialized.")
print("Current Chat History:", chat_history.messages)


model = ChatOpenAI(model="gpt-4o-mini")

print("Start chatting with the AI, Type 'exit' to quit.")

# chat_history=[]

# system_message = SystemMessage(content="You are a helpful AI assistant.")
# chat_history.append(system_message)


while True:
    human_input=input("User: ")
    if human_input.lower() == "exit":
        break
    
    chat_history.add_user_message(human_input)
    
    ai_response= model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)
    
    print(f"AI: {ai_response.content}")

print("\n-----------------------Firestore chat history-----------------------------------\n")
print(chat_history)

    
