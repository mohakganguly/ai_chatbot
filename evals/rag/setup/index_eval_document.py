from config import EVAL_COLLECTION_NAME

from rag.vectorstore.vector_store import QdrantVectorStore

from rag.ingestion.loader import load_document
from rag.ingestion.parser import parse_documents
from rag.ingestion.cleaner import clean_documents
from rag.ingestion.metadata import enrich_metadata
from rag.chunking.chunker import chunk_documents
from rag.embedding.embedding_model import get_embedding_model


PDF_PATH = "documents/Machine_Learning_Overview.pdf"


def main():

    # -----------------------------------
    # Load and process document
    # -----------------------------------

    documents = load_document(PDF_PATH)

    documents = parse_documents(documents)

    documents = clean_documents(documents)

    documents = enrich_metadata(documents)

    chunks = chunk_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    # -----------------------------------
    # Create embeddings
    # -----------------------------------

    embedder = get_embedding_model()

    embedded_chunks = embedder.embed_documents(
        chunks
    )

    # -----------------------------------
    # Create evaluation collection
    # -----------------------------------

    vector_store = QdrantVectorStore(
        collection_name=EVAL_COLLECTION_NAME
    )

    # vector_store.create_collection()

    # -----------------------------------
    # Insert evaluation document
    # -----------------------------------

    vector_store.upsert_documents(
        embedded_chunks
    )

    print(
        f"Evaluation document indexed into "
        f"'{EVAL_COLLECTION_NAME}'"
    )


if __name__ == "__main__":
    main()