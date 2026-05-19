import io
import base64
from typing import Any, Dict, List

import fitz  # PyMuPDF
import pdfplumber
from pydantic import BaseModel


class PageContent(BaseModel):
    """Structured representation of a parsed PDF page."""
    page_num: int
    text: str
    images: List[str]  # Base64 encoded images
    tables: List[List[List[str | None]]]  # List of tables, where each table is a 2D list of strings
    metadata: Dict[str, Any]


class PDFParser:
    """
    Parser for PDF documents.
    Utilizes PyMuPDF (fitz) for text and image extraction, and pdfplumber for table extraction.
    """

    @classmethod
    def parse(cls, file_bytes: bytes) -> List[PageContent]:
        """
        Parse a PDF document from raw bytes.
        
        Args:
            file_bytes (bytes): The raw PDF file bytes.
            
        Returns:
            List[PageContent]: A list of structured page contents.
        """
        pages = []

        # We need to process via fitz for text/images and pdfplumber for tables
        # They both can load from bytes/stream
        pdf_stream = io.BytesIO(file_bytes)
        
        # Open with PyMuPDF
        with fitz.open("pdf", file_bytes) as fitz_doc:
            metadata = fitz_doc.metadata or {}
            
            # Open with pdfplumber (needs seek(0) if stream is reused, but we pass the stream directly)
            with pdfplumber.open(pdf_stream) as plumber_doc:
                
                # Iterate over pages
                for page_num in range(len(fitz_doc)):
                    fitz_page = fitz_doc[page_num]
                    plumber_page = plumber_doc.pages[page_num]
                    
                    # 1. Extract Text
                    text = fitz_page.get_text()
                    
                    # 2. Extract Images (convert to Base64)
                    images_b64 = []
                    image_list = fitz_page.get_images(full=True)
                    for img in image_list:
                        xref = img[0]
                        base_image = fitz_doc.extract_image(xref)
                        if base_image:
                            image_bytes = base_image["image"]
                            b64 = base64.b64encode(image_bytes).decode("utf-8")
                            images_b64.append(b64)
                            
                    # 3. Extract Tables
                    tables = plumber_page.extract_tables()
                    
                    page_content = PageContent(
                        page_num=page_num + 1,
                        text=text.strip(),
                        images=images_b64,
                        tables=tables,
                        metadata=metadata
                    )
                    pages.append(page_content)
                    
        return pages
