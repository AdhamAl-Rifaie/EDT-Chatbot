import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, Depends, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Generator, Optional
from pydantic import BaseModel
import asyncio
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage, AIMessage
from deepface import DeepFace
from io import BytesIO

app = FastAPI()
# app.mount("/static", StaticFiles(directory="EDT-ChatBot"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# Models
class ChatRequest(BaseModel):
    user_question: str
    session_id: str
    emotion: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str

class ChatHistoryResponse(BaseModel):
    history: List[Message]

class EmotionChatResponse(BaseModel):
    emotion: str
    response: str

# Helper functions
def read_system_prompt(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()

def enhance_image(image_bytes):
    # Read image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Apply CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img)
    
    # Convert back to BGR for DeepFace
    return cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

# Chatbot response generation
async def stream_response(user_question: str, system_prompt: str, session_id: str, emotion: Optional[str] = None) -> Generator[str, None, None]:
    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature=0.3,
        api_key="97a57dd4999fb6c50550baf7d0e4c7747537c42512417deb6afa28280975c438"
    )

    # If emotion is detected, add it to the prompt
    if emotion:
        enhanced_prompt = system_prompt + f"\nThe user's current detected emotion is: {emotion}. Please respond appropriately considering this emotional state."
    else:
        enhanced_prompt = system_prompt

    prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_prompt),
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

# Routes
@app.get("/")
async def get_homepage():
    return FileResponse("EDT-ChatBot/Main-Page.html")

# Original chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    system_prompt = read_system_prompt("utils/system_prompt.txt")
    return StreamingResponse(
        stream_response(request.user_question, system_prompt, request.session_id, request.emotion), 
        media_type="text/plain"
    )

# Emotion detection endpoint
@app.post("/detect-emotion/")
async def detect_emotion(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    try:
        enhanced_img = enhance_image(image_bytes)
        
        result = DeepFace.analyze(
            img_path=enhanced_img, 
            actions=['emotion'], 
            enforce_detection=False, 
            detector_backend='opencv'
        )
        
        return {
            "dominant_emotion": result[0]['dominant_emotion'],
            "emotion_scores": result[0]['emotion']
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing image: {str(e)}"}
        )

# New endpoint that combines emotion detection and chat
@app.post("/emotion-chat/")
async def emotion_chat(
    file: UploadFile = File(...),
    user_question: str = Form(...),
    session_id: str = Form(...)
):
    # First, detect emotion
    image_bytes = await file.read()
    
    try:
        enhanced_img = enhance_image(image_bytes)
        
        result = DeepFace.analyze(
            img_path=enhanced_img, 
            actions=['emotion'], 
            enforce_detection=False, 
            detector_backend='opencv'
        )
        
        emotion = result[0]['dominant_emotion']
        
        # Then, get chatbot response with emotion context
        system_prompt = read_system_prompt("utils/system_prompt.txt")
        
        # Create a collector for the streamed response
        response_chunks = []
        async for chunk in stream_response(user_question, system_prompt, session_id, emotion):
            response_chunks.append(chunk)
        
        full_response = "".join(response_chunks)
        
        # Save to chat history (optional - this might duplicate if stream_response already does it)
        history = get_memory(session_id)
        history.add_user_message(user_question)
        history.add_ai_message(full_response)
        
        return EmotionChatResponse(
            emotion=emotion,
            response=full_response
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing request: {str(e)}"}
        )

# Chat history endpoint
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