from db.database import engine
from db.models import Base

from db.checkpoint import checkpointer


def initialize_database():

    print("Creating application tables...")

    Base.metadata.create_all(bind=engine)

    print("Application tables created.")

    print("Creating LangGraph checkpoint tables...")

    checkpointer.setup()

    print("Checkpoint tables created.")

    print("Database initialization complete.")


if __name__ == "__main__":
    initialize_database()