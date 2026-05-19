from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.services.rag_service.src.infrastructure.chunkers.base_chunker import BaseChunker, ChunkPair


class ParentChildChunker(BaseChunker):
    """
    Industrial Parent-Child Chunker.
    
    Rationale: Industrial maintenance manuals and SOPs have long, complex procedures 
    (the "parent" context) but engineers often query for exact parameter values or single 
    action steps (the "child" retrieval target).
    
    This strategy chunks the document into large parent contexts (stored in a fast KV store 
    like Redis) and then sub-chunks those into small, highly-specific child chunks (embedded 
    in a Vector DB like Qdrant). When a child matches a query, the LLM is fed the entire 
    parent context to prevent hallucination of the broader procedure.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        parent_overlap: int = 200,
        child_chunk_size: int = 300,
        child_overlap: int = 50,
    ) -> None:
        super().__init__()
        
        separators = ["\n\n", "\n", ".", " "]
        
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            length_function=self.count_tokens,
            separators=separators,
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            length_function=self.count_tokens,
            separators=separators,
        )

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[ChunkPair]:
        """
        Splits text into parent-child ChunkPairs.
        
        Args:
            text (str): The raw document text.
            metadata (Dict[str, Any]): Base metadata including 'doc_id'.
            
        Returns:
            List[ChunkPair]: The hierarchical chunk pairs.
        """
        doc_id = metadata.get("doc_id", "unknown_doc")
        pairs: List[ChunkPair] = []
        
        parent_docs = self.parent_splitter.create_documents([text])
        
        for p_idx, p_doc in enumerate(parent_docs):
            parent_id = self.generate_chunk_id(doc_id, p_idx)
            parent_text = p_doc.page_content
            
            # Sub-chunk the parent into children
            child_docs = self.child_splitter.create_documents([parent_text])
            
            for c_idx, c_doc in enumerate(child_docs):
                child_id = self.generate_chunk_id(parent_id, c_idx)
                
                # Combine metadata securely
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    "parent_id": parent_id,
                    "chunk_strategy": "parent_child"
                })
                
                pairs.append(
                    ChunkPair(
                        parent_id=parent_id,
                        parent_text=parent_text,
                        child_id=child_id,
                        child_text=c_doc.page_content,
                        metadata=chunk_meta
                    )
                )
                
        return pairs
