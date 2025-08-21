from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
from .models import QuestionRequest, AnswerResponse, DocumentMetadata
from .database import db
from .vector_store import vector_store
from .document_processor import doc_processor
from .rag_chain import rag_chain

app = FastAPI(
    title="Conversational RAG Q&A System",
    description="A RAG system with document upload, multi-turn conversations, and reasoning",
    version="1.0.0"
)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (.pdf or .txt)"""
    
    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(
            status_code=400, 
            detail="Only .pdf and .txt files are supported"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Process document
        metadata, chunks = await doc_processor.process_document(file.filename, content)
        
        # Store in MongoDB
        await db.store_document_metadata(metadata)
        await db.store_document_chunks(chunks)
        
        # Add to vector store
        await vector_store.add_chunks(chunks)
        
        return {
            "message": "Document uploaded successfully",
            "document_id": metadata.document_id,
            "filename": metadata.filename,
            "total_chunks": metadata.total_chunks,
            "total_pages": metadata.total_pages
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer with references and reasoning"""
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        response = await rag_chain.answer_question(
            user_id=request.user_id,
            question=request.question,
            top_k=request.top_k
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/history")
async def get_history(user_id: str = Query(..., description="User ID")):
    """Get conversation history for a user"""
    
    try:
        history = await db.get_conversation_history(user_id)
        return {
            "user_id": user_id,
            "conversation_history": history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    """List all uploaded documents with metadata"""
    
    try:
        documents = await db.get_documents_list()
        return documents
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.delete("/clear")
async def clear_system():
    """Clear all data from the system"""
    
    try:
        # Clear MongoDB collections
        await db.clear_all_data()
        
        # Clear vector store
        await vector_store.clear_collection()
        
        return {"message": "System cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing system: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Conversational RAG Q&A System",
        "version": "1.0.0",
        "endpoints": {
            "/upload": "POST - Upload documents (.pdf, .txt)",
            "/ask": "POST - Ask questions",
            "/history": "GET - Get conversation history",
            "/documents": "GET - List uploaded documents",
            "/clear": "DELETE - Clear all data"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        docs_count = len(await db.get_documents_list())
        
        return {
            "status": "healthy",
            "database": "connected",
            "documents_count": docs_count
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    