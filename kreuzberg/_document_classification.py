from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final
from PIL import Image
from transformers import pipeline

from kreuzberg._types import ExtractionConfig, ExtractionResult
from kreuzberg._ocr._tesseract import PSMMode, TesseractConfig


class DocumentType(Enum):
    ADVERTISEMENT = "advertisement"
    HANDWRITTEN = "handwritten"
    INVOICE = "invoice" 
    LETTER = "letter"
    NEWS_ARTICLE = "news_article"
    RESUME = "resume"
    UNKNOWN = "unknown"


@dataclass
class DocumentClassification:
    document_type: DocumentType
    confidence: float


class DocumentClassifier:
    def __init__(self):
        self.model_name = "prithivMLmods/Document-Type-Detection"
        self.pipeline = pipeline(
            task="image-classification",
            model=self.model_name
        )
        
        # Map labels to DocumentType enum
        self.label2type = {
            "Advertisement-Doc": DocumentType.ADVERTISEMENT,
            "Hand-Written-Doc": DocumentType.HANDWRITTEN,
            "Invoice-Doc": DocumentType.INVOICE,
            "Letter-Doc": DocumentType.LETTER,
            "News-Article-Doc": DocumentType.NEWS_ARTICLE,
            "Resume-Doc": DocumentType.RESUME
        }

    def classify_image(self, image: Image.Image) -> DocumentClassification:
        """Classify document image using the pipeline."""
        try:
            # Get predictions from pipeline
            predictions = self.pipeline(image)
            if not predictions:
                return DocumentClassification(DocumentType.UNKNOWN, 0.0)
                
            # Get top prediction
            top_pred = predictions[0]
            doc_type = self.label2type.get(top_pred['label'], DocumentType.UNKNOWN)
            confidence = float(top_pred['score'])
            
            return DocumentClassification(doc_type, confidence)
            
        except Exception as e:
            print(f"Classification error: {e}")
            return DocumentClassification(DocumentType.UNKNOWN, 0.0)

