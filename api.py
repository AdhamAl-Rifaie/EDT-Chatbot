from fastapi import FastAPI, WebSocket, Depends
from fastapi.responses import StreamingResponse
from typing import List, Generator
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, AIMessage


app = FastAPI()
app.mount("/static", StaticFiles(directory="EDT-ChatBot"), name="static")

# Optional: Serve the main HTML file at root
@app.get("/")
async def get_homepage():
    return FileResponse("EDT-ChatBot/Main-Page.html")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace * with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# In-memory session storage
session_memory = {}

def get_memory(session_id: str):
    if session_id not in session_memory:
        session_memory[session_id] = ChatMessageHistory()
    return session_memory[session_id]

# Updated ChatRequest model without chat_history
class ChatRequest(BaseModel):
    user_question: str
    session_id: str

# Response models for chat history endpoint
class Message(BaseModel):
    role: str
    content: str

class ChatHistoryResponse(BaseModel):
    history: List[Message]

def read_system_prompt(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()

# Updated stream_response without chat_history
async def stream_response(user_question: str, system_prompt: str, session_id: str) -> Generator[str, None, None]:
    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature=0.3,
        api_key="97a57dd4999fb6c50550baf7d0e4c7747537c42512417deb6afa28280975c438"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    history = get_memory(session_id)
    
    chain = prompt | llm
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    async for chunk in chain_with_memory.astream(
        {"input": user_question},
        config={"configurable": {"session_id": session_id}}
    ):
        yield chunk.content

# Updated /chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    system_prompt = read_system_prompt("utils/system_prompt.txt")
    return StreamingResponse(stream_response(request.user_question, system_prompt, request.session_id), media_type="text/plain")

# New endpoint to retrieve chat history
@app.post("/chat_history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    history = get_memory(session_id)
    messages = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            messages.append(Message(role="user", content=msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(Message(role="assistant", content=msg.content))
    return ChatHistoryResponse(history=messages)