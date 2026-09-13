from qdrant_client import QdrantClient, models

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
)


THREAD_ID = "08abecdc-451c-4b3a-b370-b450b0ba33b6"


def main():

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="thread_id",
                match=models.MatchValue(
                    value=THREAD_ID,
                ),
            )
        ]
    )

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=query_filter,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )

    print(f"\nRetrieved {len(points)} points\n")

    for i, point in enumerate(points, start=1):
        print("=" * 80)
        print(f"Point {i}")
        print("ID:", point.id)
        print("Thread ID:", point.payload.get("thread_id"))
        print("Filename:", point.payload.get("filename"))
        print("Text:", point.payload.get("text", "")[:300])


if __name__ == "__main__":
    main()