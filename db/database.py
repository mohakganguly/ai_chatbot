from sqlalchemy import create_engine
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
import os

load_dotenv()

DATABASE_URL=os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    echo=True
)

session_local= sessionmaker(
bind=engine,
    autoflush=False,
    autocommit=False
)


