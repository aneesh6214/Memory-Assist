from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from memory_system import MemorySystem
import os
import tempfile
import json

app = FastAPI(
    title="LLM Notepad API",
    description="AI-powered second brain with vector memory storage (text and audio)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory system
memory_system = MemorySystem()

# Request/Response models
class StoreMemoryRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class StoreMemoryResponse(BaseModel):
    memory_ids: List[str]
    chunks_created: int
    message: str
    transcription: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    relevant_memories: List[Dict[str, Any]]
    transcription: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5

class SearchResponse(BaseModel):
    memories: List[Dict[str, Any]]

async def transcribe_audio(audio_file: UploadFile) -> str:
    """Transcribe audio file using OpenAI Whisper."""
    try:
        # Save uploaded file temporarily with original extension
        file_extension = ".webm" if "webm" in audio_file.content_type else ".mp3"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe using OpenAI Whisper
        with open(temp_file_path, "rb") as audio:
            transcript = memory_system.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                response_format="text"
            )
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return transcript.strip()
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "LLM Notepad API - Your AI-powered second brain (text and audio)"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llm-notepad"}

# TEXT ENDPOINTS
@app.post("/text/store", response_model=StoreMemoryResponse)
async def store_text_memory(request: StoreMemoryRequest):
    """Store a text memory in the vector database."""
    try:
        memory_ids = memory_system.store_memory(request.text, request.metadata)
        return StoreMemoryResponse(
            memory_ids=memory_ids,
            chunks_created=len(memory_ids),
            message=f"Successfully stored text memory in {len(memory_ids)} chunks"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store text memory: {str(e)}")

@app.get("/text/query", response_model=QueryResponse)
async def query_text_memories(query: str, max_context_memories: int = 3):
    """Query memories using text input with LLM context."""
    try:
        # Get relevant memories first
        relevant_memories = memory_system.search_memories(query, max_context_memories)
        
        # Get LLM response with context
        response = memory_system.query_with_context(query, max_context_memories)
        
        return QueryResponse(
            response=response,
            relevant_memories=relevant_memories
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query text memories: {str(e)}")

@app.post("/text/search", response_model=SearchResponse)
async def search_text_memories(request: SearchRequest):
    """Search for memories using text without LLM processing (raw results)."""
    try:
        memories = memory_system.search_memories(request.query, request.n_results)
        return SearchResponse(memories=memories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search text memories: {str(e)}")

# AUDIO ENDPOINTS
@app.post("/audio/store", response_model=StoreMemoryResponse)
async def store_audio_memory(
    audio_file: UploadFile = File(..., description="Audio file to transcribe and store (webm, mp3, wav, m4a, flac)"),
    metadata: Optional[str] = None
):
    """Store an audio memory by transcribing it first."""
    try:
        # Validate file type
        allowed_types = ["audio/webm", "audio/mp3", "audio/mpeg", "audio/wav", "audio/m4a", "audio/flac"]
        if audio_file.content_type not in allowed_types and not any(
            audio_file.filename.lower().endswith(ext) for ext in ['.webm', '.mp3', '.wav', '.m4a', '.flac']
        ):
            raise HTTPException(
                status_code=400, 
                detail="Only audio files (webm, mp3, wav, m4a, flac) are supported"
            )
        
        # Transcribe audio
        transcription = await transcribe_audio(audio_file)
        
        if not transcription:
            raise HTTPException(status_code=400, detail="No speech detected in audio file")
        
        # Parse metadata if provided
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except:
                parsed_metadata = {"note": metadata}
        
        # Add transcription metadata
        parsed_metadata["source"] = "audio"
        parsed_metadata["original_filename"] = audio_file.filename
        parsed_metadata["content_type"] = audio_file.content_type
        
        # Store transcribed text
        memory_ids = memory_system.store_memory(transcription, parsed_metadata)
        
        # Log the storage with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"🗄️  [{timestamp}] Stored memory: {len(memory_ids)} chunks")
        print(f"📝 [{timestamp}] Transcription: {transcription[:100]}{'...' if len(transcription) > 100 else ''}")
        
        return StoreMemoryResponse(
            memory_ids=memory_ids,
            chunks_created=len(memory_ids),
            message=f"Successfully transcribed and stored audio memory in {len(memory_ids)} chunks",
            transcription=transcription
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store audio memory: {str(e)}")

@app.post("/audio/query", response_model=QueryResponse)
async def query_audio_memories(
    audio_file: UploadFile = File(..., description="Audio file with query (webm, mp3, wav, m4a, flac)"),
    max_context_memories: int = Form(3)
):
    """Query memories using audio input (transcribed to text first)."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"🤖 [{timestamp}] Received query request - file: {audio_file.filename}, size: {audio_file.size if hasattr(audio_file, 'size') else 'unknown'}")
    
    try:
        # Validate file type
        allowed_types = ["audio/webm", "audio/mp3", "audio/mpeg", "audio/wav", "audio/m4a", "audio/flac"]
        if audio_file.content_type not in allowed_types and not any(
            audio_file.filename.lower().endswith(ext) for ext in ['.webm', '.mp3', '.wav', '.m4a', '.flac']
        ):
            raise HTTPException(
                status_code=400, 
                detail="Only audio files (webm, mp3, wav, m4a, flac) are supported"
            )
        
        # Transcribe audio query
        query_text = await transcribe_audio(audio_file)
        
        if not query_text:
            raise HTTPException(status_code=400, detail="No speech detected in audio file")
        
        # Get relevant memories
        relevant_memories = memory_system.search_memories(query_text, max_context_memories)
        
        # Get LLM response with context
        response = memory_system.query_with_context(query_text, max_context_memories)
        
        return QueryResponse(
            response=response,
            relevant_memories=relevant_memories,
            transcription=query_text
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query audio memories: {str(e)}")

@app.get("/memories/count")
async def get_memory_count():
    """Get the total number of stored memory chunks."""
    try:
        count = memory_system.collection.count()
        return {"total_memories": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory count: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)