import PyPDF2
import tiktoken
import uuid
from typing import List, Tuple
from datetime import datetime
from io import BytesIO
from .config import config
from .models import DocumentChunk, DocumentMetadata

class DocumentProcessor:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 encoding
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def extract_text_from_pdf(self, file_content: bytes) -> Tuple[str, int]:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text_pages = []
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                text_pages.append((text, page_num))
            
            # Combine all text
            full_text = "\n".join([page[0] for page in text_pages])
            total_pages = len(pdf_reader.pages)
            
            return full_text, total_pages
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")
    
    def extract_text_from_txt(self, file_content: bytes) -> Tuple[str, int]:
        """Extract text from TXT file"""
        try:
            text = file_content.decode('utf-8')
            return text, 1  # TXT files have 1 logical page
        except UnicodeDecodeError:
            try:
                text = file_content.decode('latin-1')
                return text, 1
            except Exception as e:
                raise ValueError(f"Error processing TXT file: {str(e)}")
    
    def chunk_text(self, text: str, document_id: str, filename: str, page_mapping: dict = None) -> List[DocumentChunk]:
        """Chunk text into smaller pieces with overlap"""
        chunks = []
        chunk_size = config.CHUNK_SIZE
        overlap = config.CHUNK_OVERLAP
        
        # Split text into sentences for better chunking
        sentences = text.split('.')
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence += "."  # Add back the period
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, create a chunk
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Create chunk
                chunk_id = f"{document_id}_chunk_{chunk_index}"
                page_num = self._get_page_for_text(current_chunk, page_mapping) if page_mapping else None
                
                chunk = DocumentChunk(
                    document_id=document_id,
                    filename=filename,
                    chunk_id=chunk_id,
                    chunk_text=current_chunk.strip(),
                    page_number=page_num,
                    upload_timestamp=datetime.utcnow()
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_index += 1
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            page_num = self._get_page_for_text(current_chunk, page_mapping) if page_mapping else None
            
            chunk = DocumentChunk(
                document_id=document_id,
                filename=filename,
                chunk_id=chunk_id,
                chunk_text=current_chunk.strip(),
                page_number=page_num,
                upload_timestamp=datetime.utcnow()
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last 'overlap_tokens' worth of text"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        
        overlap_token_list = tokens[-overlap_tokens:]
        return self.encoding.decode(overlap_token_list)
    
    def _get_page_for_text(self, text: str, page_mapping: dict) -> int:
        """Determine which page a chunk of text belongs to"""
        # This is a simplified implementation
        # In a real-world scenario, you'd want more sophisticated page tracking
        return 1
    
    async def process_document(self, filename: str, file_content: bytes) -> Tuple[DocumentMetadata, List[DocumentChunk]]:
        """Process a document and return metadata and chunks"""
        document_id = str(uuid.uuid4())
        file_size = len(file_content)
        file_type = filename.split('.')[-1].lower()
        
        # Extract text based on file type
        if file_type == 'pdf':
            full_text, total_pages = self.extract_text_from_pdf(file_content)
        elif file_type == 'txt':
            full_text, total_pages = self.extract_text_from_txt(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Create chunks
        chunks = self.chunk_text(full_text, document_id, filename)
        
        # Create metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            upload_timestamp=datetime.utcnow(),
            total_chunks=len(chunks),
            total_pages=total_pages
        )
        
        return metadata, chunks

# Global document processor instance
doc_processor = DocumentProcessor()