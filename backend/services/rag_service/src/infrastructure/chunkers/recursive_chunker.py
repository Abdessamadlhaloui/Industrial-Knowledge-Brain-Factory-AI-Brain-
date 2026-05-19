from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.services.rag_service.src.infrastructure.chunkers.base_chunker import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    """
    Standard recursive text splitter.
    
    Focuses on preserving markdown structure (headers, code blocks, tables).
    Used as the fast path for simple, cleanly structured documents.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        super().__init__()
        
        # Markdown specific separators to preserve block structures
        separators = [
            "\n#{1,6} ",  # Headers
            "```\n",      # Code blocks
            "\n\n***\n\n",# Horizontal rules
            "\n\n",       # Paragraphs
            "\n",         # Lines
            ". ",         # Sentences
            " ",          # Words
            ""            # Characters
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.count_tokens,
            separators=separators,
        )

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Splits text recursively based on Markdown structure.
        """
        doc_id = metadata.get("doc_id", "unknown_doc")
        chunks: List[Chunk] = []
        
        docs = self.splitter.create_documents([text])
        
        # We need to approximate start/end idx since LangChain doesn't provide it directly in create_documents
        current_idx = 0
        
        for idx, doc in enumerate(docs):
            chunk_text = doc.page_content
            chunk_meta = metadata.copy()
            chunk_meta["chunk_strategy"] = "recursive"
            
            # Extremely rough approximation of start_idx
            start_idx = text.find(chunk_text, current_idx)
            if start_idx == -1:
                start_idx = current_idx
            
            chunks.append(
                Chunk(
                    chunk_id=self.generate_chunk_id(doc_id, idx),
                    text=chunk_text,
                    metadata=chunk_meta,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    token_count=self.count_tokens(chunk_text)
                )
            )
            current_idx = start_idx + len(chunk_text) // 2  # advance roughly

        return chunks
