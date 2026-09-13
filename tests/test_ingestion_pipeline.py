from rag.ingestion.loader import load_document
from rag.ingestion.parser import parse_documents
from rag.ingestion.cleaner import clean_documents
from rag.ingestion.metadata import enrich_metadata
from rag.chunking.chunker import chunk_documents

pdf = "documents/resume_flipkart.pdf"

docs = load_document(pdf)
print("Loader ✓")

docs = parse_documents(docs)
print("Parser ✓")

docs = clean_documents(docs)
print("Cleaner ✓")

docs = enrich_metadata(docs)
print("Metadata ✓")

chunks = chunk_documents(docs)
print("Chunker ✓")

print(f"Chunks created: {len(chunks)}")

print("\nFirst chunk metadata:")
print(chunks[0].metadata)

print("\nFirst chunk:")
print(chunks[0].page_content[:500])