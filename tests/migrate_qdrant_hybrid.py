"""
One-time Qdrant migration.

WARNING:
This deletes the existing collection and all vectors inside it.

After running this script, documents must be re-indexed.
"""

from rag.vectorstore.vector_store import QdrantVectorStore
from utils.logger import get_logger


logger = get_logger(__name__)


def main():

    logger.info("=" * 80)
    logger.info("Starting Qdrant hybrid migration")
    logger.info("=" * 80)

    store = QdrantVectorStore()

    # ------------------------------------------------------
    # WARNING
    # ------------------------------------------------------

    logger.warning(
        "The existing Qdrant collection will be deleted."
    )

    store.recreate_collection()

    # ------------------------------------------------------
    # Verify
    # ------------------------------------------------------

    info = store.collection_info()

    logger.info(
        "New collection information:\n%s",
        info,
    )

    logger.info("=" * 80)
    logger.info("Qdrant hybrid migration completed")
    logger.info("Documents must now be re-indexed.")
    logger.info("=" * 80)


if __name__ == "__main__":

    main()