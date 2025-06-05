from _document_classification import DocumentClassifier

clf = DocumentClassifier()
doc_type, confidence = clf.classify(r"C:\Users\drivi\Downloads\final phase 2report.docx")