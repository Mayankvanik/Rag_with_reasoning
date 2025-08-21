from pymongo import MongoClient
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from .config import config
from .models import DocumentChunk, ConversationTurn, UserHistory, DocumentMetadata

class MongoDB:
    def __init__(self):
        self.client = MongoClient(config.MONGODB_URL)
        self.db = self.client[config.DATABASE_NAME]
        self.documents = self.db.documents
        self.chunks = self.db.chunks
        self.conversations = self.db.conversations
        
    async def store_document_metadata(self, metadata: DocumentMetadata) -> str:
        """Store document metadata"""
        result = self.documents.insert_one(metadata.dict())
        return str(result.inserted_id)
    
    async def store_document_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Store document chunks"""
        chunk_dicts = [chunk.dict() for chunk in chunks]
        result = self.chunks.insert_many(chunk_dicts)
        return [str(id) for id in result.inserted_ids]
    
    async def get_conversation_history(self, user_id: str, limit: int = None) -> List[ConversationTurn]:
        """Get user's conversation history"""
        if limit is None:
            limit = config.MAX_HISTORY_TURNS
            
        history = self.conversations.find_one({"user_id": user_id})
        if not history:
            return []
        
        conversations = history.get("conversations", [])
        # Return last 'limit' conversations
        return [ConversationTurn(**conv) for conv in conversations[-limit:]]
    
    async def store_conversation_turn(self, user_id: str, turn: ConversationTurn) -> str:
        """Store a conversation turn"""
        # Update or create user conversation history
        self.conversations.update_one(
            {"user_id": user_id},
            {
                "$push": {"conversations": turn.dict()},
                "$set": {"last_updated": datetime.utcnow()}
            },
            upsert=True
        )
        return user_id
    
    async def get_documents_list(self) -> List[DocumentMetadata]:
        """Get list of all documents"""
        docs = list(self.documents.find())
        return [DocumentMetadata(**doc) for doc in docs]
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get chunk by ID"""
        chunk = self.chunks.find_one({"chunk_id": chunk_id})
        if chunk:
            return DocumentChunk(**chunk)
        return None
    
    async def clear_all_data(self):
        """Clear all data from collections"""
        self.documents.delete_many({})
        self.chunks.delete_many({})
        self.conversations.delete_many({})

# Global database instance
db = MongoDB()