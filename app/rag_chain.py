import openai
from typing import List, Tuple, Dict, Any
from datetime import datetime
from .config import config
from .models import ConversationTurn, Reference, AnswerResponse
from .vector_store import vector_store
from .database import db

class RAGChain:
    def __init__(self):
        openai.api_key = config.OPENAI_API_KEY
    
    def _build_context_prompt(self, 
                            question: str, 
                            history: List[ConversationTurn], 
                            retrieved_chunks: List[Tuple[str, str, Dict, float]]) -> str:
        """Build the RAG prompt with context"""
        
        prompt_parts = ["You are a helpful assistant that answers questions based on provided documents."]
        
        # Add conversation history if available
        if history:
            prompt_parts.append("\n## Previous Conversation:")
            for turn in history[-3:]:  # Last 3 turns
                prompt_parts.append(f"Q: {turn.question}")
                prompt_parts.append(f"A: {turn.answer}")
        
        # Add retrieved document context
        if retrieved_chunks:
            prompt_parts.append("\n## Document Context:")
            for i, (chunk_id, content, metadata, score) in enumerate(retrieved_chunks):
                filename = metadata.get('filename', 'Unknown')
                page_num = metadata.get('page_number', 'N/A')
                prompt_parts.append(f"\n[Document {i+1}: {filename}, Page {page_num}]")
                prompt_parts.append(f"{content}")
        
        prompt_parts.append(f"\n## Current Question:\n{question}")
        
        prompt_parts.append("""
## Instructions:
1. Answer the question based ONLY on the provided document context
2. If the answer cannot be found in the documents, say so clearly
3. Cite specific documents and pages when possible
4. Provide reasoning for how you arrived at your answer
5. Be concise but comprehensive
6. Consider the conversation history for context

Please provide your answer in the following format:
ANSWER: [Your detailed answer here]
REASONING: [Explain how you derived this answer from the documents]
""")
        
        return "\n".join(prompt_parts)
    
    def _generate_suggestions(self, question: str, answer: str, references: List[Reference]) -> List[str]:
        """Generate follow-up question suggestions"""
        suggestions = []
        
        # Extract key topics from references
        topics = set()
        for ref in references:
            # Simple keyword extraction (could be improved with NLP)
            words = ref.content_snippet.lower().split()
            topics.update([word for word in words if len(word) > 4])
        
        # Generate contextual suggestions based on the current question and topics
        if "what" in question.lower():
            suggestions.append(f"How does this relate to other aspects mentioned in the documents?")
        if "why" in question.lower():
            suggestions.append(f"What are the implications of this?")
        if "when" in question.lower():
            suggestions.append(f"What happened before or after this?")
        
        # Add topic-specific suggestions
        for topic in list(topics)[:2]:  # Limit to 2 topics
            suggestions.append(f"Tell me more about {topic}")
        
        return suggestions[:3]  # Return max 3 suggestions
    
    def _extract_references(self, 
                          retrieved_chunks: List[Tuple[str, str, Dict, float]], 
                          answer: str) -> List[Reference]:
        """Extract references from retrieved chunks"""
        references = []
        
        for chunk_id, content, metadata, score in retrieved_chunks:
            # Create a content snippet (first 150 characters)
            snippet = content[:150] + "..." if len(content) > 150 else content
            
            reference = Reference(
                document=metadata.get('filename', 'Unknown'),
                page=metadata.get('page_number'),
                chunk_id=chunk_id,
                content_snippet=snippet,
                relevance_score=round(score, 3)
            )
            references.append(reference)
        
        return references
    
    async def answer_question(self, 
                            user_id: str, 
                            question: str, 
                            top_k: int = 4) -> AnswerResponse:
        """Main RAG pipeline to answer a question"""
        
        # 1. Retrieve conversation history
        history = await db.get_conversation_history(user_id)
        
        # 2. Retrieve relevant chunks
        retrieved_chunks = await vector_store.similarity_search(question, top_k)
        
        if not retrieved_chunks:
            return AnswerResponse(
                answer="I couldn't find relevant information in the uploaded documents to answer your question.",
                reasoning="No documents were found that match your query. Please ensure you have uploaded relevant documents.",
                references=[],
                suggestions=["Try rephrasing your question", "Upload relevant documents first"],
                conversation_id=user_id
            )
        
        # 3. Build RAG prompt
        prompt = self._build_context_prompt(question, history, retrieved_chunks)
        
        # 4. Call OpenAI LLM
        try:
            response = openai.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful document analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            llm_response = response.choices[0].message.content
            
            # 5. Parse the response
            answer, reasoning = self._parse_llm_response(llm_response)
            
        except Exception as e:
            return AnswerResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                reasoning="There was a technical issue with the language model.",
                references=[],
                suggestions=["Please try asking your question again"],
                conversation_id=user_id
            )
        
        # 6. Extract references
        references = self._extract_references(retrieved_chunks, answer)
        
        # 7. Generate suggestions
        suggestions = self._generate_suggestions(question, answer, references)
        
        # 8. Store conversation turn
        conversation_turn = ConversationTurn(
            question=question,
            answer=answer,
            references=references,
            timestamp=datetime.utcnow()
        )
        await db.store_conversation_turn(user_id, conversation_turn)
        
        return AnswerResponse(
            answer=answer,
            reasoning=reasoning,
            references=references,
            suggestions=suggestions,
            conversation_id=user_id
        )
    
    def _parse_llm_response(self, response: str) -> Tuple[str, str]:
        """Parse the LLM response to extract answer and reasoning"""
        answer = ""
        reasoning = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("ANSWER:"):
                current_section = "answer"
                answer = line[7:].strip()
            elif line.startswith("REASONING:"):
                current_section = "reasoning"
                reasoning = line[10:].strip()
            elif current_section == "answer" and line:
                answer += " " + line
            elif current_section == "reasoning" and line:
                reasoning += " " + line
        
        # Fallback if parsing fails
        if not answer and not reasoning:
            answer = response
            reasoning = "Answer derived from document analysis."
        elif not reasoning:
            reasoning = "Based on the provided document context."
        
        return answer.strip(), reasoning.strip()

# Global RAG chain instance
rag_chain = RAGChain()