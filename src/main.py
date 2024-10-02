import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.rag_system import RagSystem

app = FastAPI()
rag_system = RagSystem("./documents", host="http://chnaaam.com:11434")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


# Define a Pydantic model for the request body
class ChatRequest(BaseModel):
    query: str


# chat : body -> {"query": "LED 좀 꺼줘 :)"}
@app.post("/")
async def chat(request: ChatRequest):
    return StreamingResponse(rag_system(request.query), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
