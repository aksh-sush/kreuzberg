import asyncio
import argparse
from pathlib import Path
import nltk
from collections import Counter
import re
from typing import Dict, List
from kreuzberg import extract_file, ExtractionConfig, TesseractConfig
from kreuzberg._document_classification import DocumentClassifier, DocumentType

def ensure_nltk_data():
    """Ensure required NLTK data is available."""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/stopwords', 'stopwords'),
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('corpora/words', 'words')
    ]
    
    for data_path, data_name in required_data:
        try:
            nltk.data.find(data_path)
        except LookupError:
            print(f"Downloading {data_name}...")
            nltk.download(data_name, quiet=True)

class DocumentAnalyzer:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.document_patterns = {
            DocumentType.INVOICE: [
                r'invoice', r'bill', r'amount', r'payment', r'due date',
                r'tax', r'total', r'subtotal', r'quantity', r'price'
            ],
            DocumentType.LETTER: [
                r'dear\s+\w+', r'sincerely', r'regards', r'yours truly',
                r'to whom it may concern', r'dear sir/madam'
            ],
            DocumentType.RESUME: [
                r'experience', r'education', r'skills', r'objective',
                r'employment', r'qualification', r'reference'
            ],
            DocumentType.NEWS_ARTICLE: [
                r'press release', r'news', r'reported', r'article',
                r'journalist', r'editor', r'published'
            ]
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = ' '.join(word for word in text.split() 
                       if word not in self.stop_words)
        return text

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract relevant features from text"""
        features = {}
        clean_text = self.clean_text(text)
        
        # Word frequency analysis
        words = nltk.word_tokenize(clean_text)
        pos_tags = nltk.pos_tag(words)
        
        # Get nouns and verbs
        nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
        verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
        
        # Named entity recognition
        named_entities = []
        for chunk in nltk.ne_chunk(pos_tags):
            if hasattr(chunk, 'label'):
                named_entities.append(' '.join(c[0] for c in chunk))

        # Calculate pattern matches for each document type
        for doc_type, patterns in self.document_patterns.items():
            pattern_matches = sum(1 for pattern in patterns 
                                if re.search(pattern, text.lower()))
            features[f"{doc_type.value}_patterns"] = pattern_matches / len(patterns)

        # Add other relevant features
        features.update({
            'noun_ratio': len(nouns) / len(words) if words else 0,
            'verb_ratio': len(verbs) / len(words) if words else 0,
            'named_entity_count': len(named_entities),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0
        })
        
        return features

async def analyze_document(file_path: str):
    """Analyze document content and classify its type"""
    try:
        # Extract text using Kreuzberg's OCR capabilities
        config = ExtractionConfig(
            force_ocr=True,
            ocr_config=TesseractConfig(language="eng")
        )
        
        # Extract text from document
        result = await extract_file(file_path, config=config)
        
        # Create analyzer instance
        analyzer = DocumentAnalyzer()
        
        # Extract features from text
        features = analyzer.extract_features(result.content)
        
        # Use DocumentClassifier for final classification
        classifier = DocumentClassifier()
        classification = await classifier.process_document(file_path)
        
        # Print detailed analysis
        print(f"\nDocument Analysis for: {Path(file_path).name}")
        print("-" * 50)
        print(f"Document Type: {classification.doc_type.value}")
        print(f"Confidence: {classification.confidence:.2%}")
        print("\nFeature Analysis:")
        for feature, value in features.items():
            print(f"{feature}: {value:.3f}")
        
        # Print extracted text sample
        print("\nExtracted Text Sample:")
        print(result.content[:500] + "...")
        
    except Exception as e:
        print(f"Error analyzing document: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Document Analysis CLI")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the document file (PDF, DOCX, or image)"
    )
    return parser.parse_args()

async def main():
    args = parse_arguments()
    file_path = args.file_path
    
    if Path(file_path).exists():
        await analyze_document(file_path)
    else:
        print(f"\nError: File not found: {file_path}")

if __name__ == "__main__":
    asyncio.run(main())