from rag.embedding.embedding_model import (
    get_embedding_model,
)

from rag.vectorstore.vector_store import (
    QdrantVectorStore,
)

from config import EVAL_COLLECTION_NAME


def main():

    query = (
        "Who coined the term machine learning "
        "and in what year?"
    )

    embedder = get_embedding_model()

    vector_store = QdrantVectorStore(
        collection_name=EVAL_COLLECTION_NAME
    )

    query_vector = embedder.embed_query(
        query
    )

    documents = vector_store.search(
        query=query,
        query_vector=query_vector,
        thread_id=None,
        limit=5,
    )

    print(
        f"\nRetrieved {len(documents)} documents\n"
    )

    for i, document in enumerate(
        documents,
        start=1,
    ):
        print("=" * 80)
        print(f"Result {i}")
        print(document.metadata)
        print()
        print(document.page_content[:500])


if __name__ == "__main__":
    main()