from PIL import Image
from transformers import pipeline

def classify_document(image_path):
    # Initialize the classification pipeline
    classifier = pipeline(
        task="image-classification",
        model="prithivMLmods/Document-Type-Detection"
    )
    
    # Load and process image
    image = Image.open(r"C:\Users\drivi\iot\images.jpeg")
    
    try:
        # Get predictions
        predictions = classifier(image)
        if predictions:
            # Get top prediction
            top_pred = predictions[0]
            doc_type = top_pred['label']
            confidence = float(top_pred['score'])
            
            print(f"Document Type: {doc_type}")
            print(f"Confidence: {confidence:.2%}")
        else:
            print("No predictions returned")
            
    except Exception as e:
        print(f"Classification error: {e}")

if __name__ == "__main__":
    # Use your image path
    classify_document("download.png")