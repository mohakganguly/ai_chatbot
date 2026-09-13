from qdrant_client import QdrantClient

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
)


def main():
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    print(f"\nFound {len(points)} points\n")

    for i, point in enumerate(points, start=1):
        payload = point.payload or {}

        print("=" * 80)
        print(f"POINT {i}")
        print("Point ID:", point.id)
        print("Thread ID:", payload.get("thread_id"))
        print("Source:", payload.get("source"))
        print("Filename:", payload.get("filename"))
        print("Text preview:")
        print(payload.get("text", "")[:200])


if __name__ == "__main__":
    main()