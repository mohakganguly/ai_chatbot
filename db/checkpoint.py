# import os
# from dotenv import load_dotenv

# from langgraph.checkpoint.postgres import PostgresSaver
# import psycopg

# load_dotenv()

# DB_URL = os.getenv("DATABASE_URL")

# conn = psycopg.connect(
#     DB_URL,
#     autocommit=True,
# )

# checkpointer = PostgresSaver(conn)

# # Creates the checkpoint tables if they don't already exist.
# checkpointer.setup()


import os

from dotenv import load_dotenv

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import psycopg

from agents.schemas import (
    ToolCall,
    ObservationDecision,
)

from tools.base import ToolResult


load_dotenv()


DB_URL = os.getenv("DATABASE_URL")


conn = psycopg.connect(
    DB_URL,
    autocommit=True,
)


# ==========================================================
# Checkpoint Serializer
# ==========================================================

serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ToolCall,
        ToolResult,
        ObservationDecision,
    ],
)


# ==========================================================
# PostgreSQL Checkpointer
# ==========================================================

checkpointer = PostgresSaver(
    conn,
    serde=serde,
)


# Creates checkpoint tables if they don't already exist.
checkpointer.setup()