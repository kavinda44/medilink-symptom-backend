import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


app = Flask(__name__)
CORS(app) 

load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("GOOGLE_API_KEY")

# AI SETUP
llm = ChatOpenAI(
    model="arcee-ai/trinity-large-preview:free", 
    openai_api_base="https://openrouter.ai/api/v1",
)

print("Loading PDF...")

loader = PyPDFLoader("medilink_chatbot.pdf") 
docs = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("Creating Vector Store with Google Embeddings...")

# Define the Google Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Create the Vector Store
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

print("System Ready!")

# CHAINS
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = (
    "You are the MediLink assistant. Use the context below to answer. "
    "IMPORTANT: Answer directly. Do NOT say 'Based on the context'.\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#SESSION MEMORY
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# API ENDPOINT
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    session_id = data.get('session_id', 'default_local')
    
    print(f"User ({session_id}): {user_message}") 

    if not user_message:
        return jsonify({"error": "No message"}), 400

    response = conversational_rag_chain.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}}
    )
    
    return jsonify({"response": response['answer']})

if __name__ == '__main__':
    print("MediLink Server running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)