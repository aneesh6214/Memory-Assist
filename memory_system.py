import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
import openai
from dotenv import load_dotenv
import tiktoken

load_dotenv()

class MemorySystem:
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("memories")
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    def chunk_text(self, text: str, max_tokens: int = 500) -> List[str]:
        """Split text into chunks based on token count."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text.strip())
            
        return chunks
    
    def store_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Store a memory by chunking it and adding to vector DB."""
        chunks = self.chunk_text(text)
        memory_ids = []
        
        for i, chunk in enumerate(chunks):
            memory_id = str(uuid.uuid4())
            memory_ids.append(memory_id)
            
            chunk_metadata = {
                "timestamp": datetime.now().isoformat(),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "original_length": len(text)
            }
            
            if metadata:
                chunk_metadata.update(metadata)
            
            self.collection.add(
                documents=[chunk],
                metadatas=[chunk_metadata],
                ids=[memory_id]
            )
        
        return memory_ids
    
    def search_memories(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant memories based on query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        memories = []
        for i in range(len(results['documents'][0])):
            memory = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            }
            memories.append(memory)
        
        return memories
    
    def query_with_context(self, user_query: str, max_context_memories: int = 3) -> str:
        """Query the LLM with relevant memory context."""
        relevant_memories = self.search_memories(user_query, max_context_memories)
        
        context = "Relevant memories:\n"
        for memory in relevant_memories:
            timestamp = memory['metadata'].get('timestamp', 'Unknown time')
            context += f"- {memory['content']} (stored at {timestamp})\n"
        
        messages = [
            {"role": "system", "content": "You are the user's personal AI memory assistant. Answer questions directly and factually using the provided context. Don't be overly conversational or mention that you're using memories - just provide the information requested. If you don't have enough information to answer the question, say so clearly."},
            {"role": "user", "content": f"{context}\n\nUser query: {user_query}"}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content