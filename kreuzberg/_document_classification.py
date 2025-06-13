from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Union
from PIL import Image
from transformers import pipeline
from docx import Document  # Add this import for DOCX handling

PDF_MIME_TYPE = "application/pdf"
DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"



class DocumentType(Enum):
    """Document types supported by the classifier."""
    ADVERTISEMENT = "advertisement"
    HANDWRITTEN = "handwritten"
    INVOICE = "invoice"
    LETTER = "letter"
    NEWS_ARTICLE = "news_article"
    RESUME = "resume"
    UNKNOWN = "unknown"

@dataclass
class DocumentClassification:
    """Result of document classification."""
    doc_type: DocumentType
    confidence: float

class DocumentClassifier:
    def __init__(self):
        self.model_name = "prithivMLmods/Document-Type-Detection"
        self.pipeline = pipeline(
            task="image-classification",
            model=self.model_name
        )

    def classify_image(self, image: Image.Image) -> DocumentClassification:
        """Classify an image using the document classification model."""
        try:
            predictions = self.pipeline(image)
            if not predictions:
                return DocumentClassification(DocumentType.UNKNOWN, 0.0)
                
            # Get top prediction
            top_pred = predictions[0]
            doc_type = top_pred['label'].lower()
            confidence = float(top_pred['score'])
            
            # Convert to DocumentType enum
            try:
                doc_type_enum = DocumentType(doc_type)
            except ValueError:
                return DocumentClassification(DocumentType.UNKNOWN, 0.0)
                
            return DocumentClassification(doc_type_enum, confidence)
            
        except Exception as e:
            print(f"Image classification error: {e}")
            return DocumentClassification(DocumentType.UNKNOWN, 0.0)

    def detect_file_type(self, file_path: Union[str, Path]) -> str:
        """Detect MIME type of the file."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        mime_types = {
            '.pdf': PDF_MIME_TYPE,
            '.docx': DOCX_MIME_TYPE,
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tiff': 'image/tiff',
            '.bmp': 'image/bmp'
        }
        return mime_types.get(suffix, 'unknown')



    async def process_document(self, file_path: Union[str, Path]) -> DocumentClassification:
        """Process document based on file type and classify content."""
        mime_type = self.detect_file_type(file_path)
        
        try:
            if mime_type in ("image/jpeg", "image/png", "image/tiff", "image/bmp"):
                # For images, use direct classification
                image = Image.open(file_path)
                return self.classify_image(image)
                
            elif mime_type == PDF_MIME_TYPE:
                # Use built-in PDF extractor with OCR support
                result = await self.pdf_extractor.extract_path_async(Path(file_path))
                return self.classify_text(result.content)
                    
            elif mime_type == DOCX_MIME_TYPE:
                # Extract text directly from DOCX
                doc = Document(file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return self.classify_text(text)
                
            return DocumentClassification(DocumentType.UNKNOWN, 0.0)
            
        except Exception as e:
            print(f"Document processing error: {e}")
            return DocumentClassification(DocumentType.UNKNOWN, 0.0)

