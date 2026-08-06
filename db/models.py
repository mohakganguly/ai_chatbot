from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    ForeignKey
)

from datetime import datetime

class Base(DeclarativeBase):
    pass

class Conversation(Base):

    __tablename__ = "conversations"

    thread_id = Column(
        String,
        primary_key=True
    )

    title = Column(
        String,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.now()
    )

    updated_at = Column(
        DateTime,
        default=datetime.now()
    )


class Message(Base):

    __tablename__ = "messages"

    id = Column(
        String,
        primary_key=True
    )

    thread_id = Column(
        String,
        ForeignKey("conversations.thread_id")
    )

    role = Column(
        String,
        nullable=False
    )

    content = Column(
        Text,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.now()
    )