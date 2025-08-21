import chromadb
from chromadb.config import Settings
from typing import List, Tuple, Dict, Any
import openai
import numpy as np
from .config import config
from .models import DocumentChunk

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIRECTORY)
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        openai.api_key = config.OPENAI_API_KEY
        
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI"""
        try:
            response = openai.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    async def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to vector store"""
        texts = [chunk.chunk_text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Get embeddings for all chunks
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)
        
        # Prepare metadata
        metadatas = []
        for chunk in chunks:
            metadata = {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number or 0,
                "upload_timestamp": chunk.upload_timestamp.isoformat()
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=chunk_ids
        )
    
    async def similarity_search(self, query: str, top_k: int = 4) -> List[Tuple[str, str, Dict, float]]:
        """Search for similar chunks"""
        query_embedding = await self.get_embedding(query)
        
        if not query_embedding:
            return []
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        search_results = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            document = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            # Convert distance to similarity score (1 - distance for cosine)
            similarity_score = 1 - results["distances"][0][i]
            
            search_results.append((chunk_id, document, metadata, similarity_score))
        
        return search_results
    
    async def clear_collection(self):
        """Clear all vectors from collection"""
        # Delete and recreate collection
        self.client.delete_collection("document_chunks")
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
# Global vector store instance
vector_store = VectorStore()