import os
from dotenv import load_dotenv

from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

conn = psycopg.connect(
    DB_URL,
    autocommit=True,
)

checkpointer = PostgresSaver(conn)

# Creates the checkpoint tables if they don't already exist.
checkpointer.setup()