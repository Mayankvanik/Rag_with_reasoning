from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentChunk(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_text: str
    page_number: Optional[int] = None
    upload_timestamp: datetime
    metadata: Optional[Dict[str, Any]] = {}

class QuestionRequest(BaseModel):
    user_id: str
    question: str
    top_k: Optional[int] = 4

class Reference(BaseModel):
    document: str
    page: Optional[int]
    chunk_id: str
    content_snippet: str
    relevance_score: float

class AnswerResponse(BaseModel):
    answer: str
    reasoning: str
    references: List[Reference]
    suggestions: List[str]
    conversation_id: str

class ConversationTurn(BaseModel):
    question: str
    answer: str
    references: List[Reference]
    timestamp: datetime

class UserHistory(BaseModel):
    user_id: str
    conversations: List[ConversationTurn]
    last_updated: datetime

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    upload_timestamp: datetime
    total_chunks: int
    total_pages: Optional[int] = None