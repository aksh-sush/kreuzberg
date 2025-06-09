import asyncio
import argparse
from pathlib import Path
from kreuzberg._document_classification import DocumentClassifier

async def classify_document(file_path: str):
    """Classify a document and print results."""
    classifier = DocumentClassifier()
    
    try:
        # Process the document
        result = await classifier.process_document(file_path)
        
        # Print results
        print(f"\nResults for: {Path(file_path).name}")
        print(f"Document Type: {result.doc_type.value}")
        print(f"Confidence: {result.confidence:.2%}")
        
    except Exception as e:
        print(f"Error classifying document: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Document Classification CLI")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the document file (PDF, DOCX, or image)"
    )
    return parser.parse_args()

async def main():
    # Get file path from command line
    args = parse_arguments()
    file_path = args.file_path
    
    if Path(file_path).exists():
        await classify_document(file_path)
    else:
        print(f"\nError: File not found: {file_path}")

if __name__ == "__main__":
    asyncio.run(main())