from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import uuid
import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Initialize the vector store and model
index = faiss.IndexFlatL2(768)  # Dimension of the embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# CORS (Cross-Origin Resource Sharing) Configuration
origins = [
    "http://localhost",
    "http://localhost:3000",  # Assuming your React app is running on this port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    threadId: str
    message: str

# In-memory thread storage
threads = {}

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def generate_llm_response(text):
    # Use your Language Model (LLM) to generate a response based on the input text
    # This could involve feeding the input text to the model and generating output
    # Replace this with the actual code for using your LLM
    
    # For example, if you're using Hugging Face's transformers library:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Tokenize input text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Generate response
    max_length = 50  # Maximum length of generated response
    response_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decode response
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return response_text

@app.post("/query")
async def query(request: QueryRequest):
    thread = threads.get(request.threadId)  # Use get() method to safely access thread object
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    content = thread.get("content")  # Get the content from the thread object
    if not content:
        raise HTTPException(status_code=404, detail="Thread content not found")

    # Decode the content if necessary (assuming it's encoded text)
    text = content.decode("utf-8")

    # Split text into chunks
    chunks = text.split("\n")

    query_embedding = embed_text(request.message)

    D, I = index.search(query_embedding, k=5)
    relevant_chunks = [chunks[i] for i in I[0]]

    # Combine chunks and query with LLM for response
    combined_text = " ".join(relevant_chunks) + " " + request.message
    
    # Generate response using LLM (replace with actual LLM code)
    response = generate_llm_response(combined_text)

    return {"response": response}

    
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        thread_id = str(uuid.uuid4())
        # Save or process the file content as needed
        threads[thread_id] = {"content": content}
        return {"threadId": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
