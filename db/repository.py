from sqlalchemy.orm import Session

from db.database import session_local
from db.models import Conversation
from db.models import Message

from datetime import datetime
import uuid

def create_conversation(
    thread_id: str,
    title: str
):
    db: Session = session_local()

    try:

        conversation = Conversation(
            thread_id=thread_id,
            title=title
        )

        db.add(conversation)

        db.commit()

    finally:

        db.close()



def save_message(
    thread_id: str,
    role: str,
    content: str
):
    db: Session = session_local()

    try:

        message = Message(

            id=str(uuid.uuid4()),

            thread_id=thread_id,

            role=role,

            content=content
        )

        db.add(message)

        db.commit()

    finally:

        db.close()


def get_messages(
    thread_id: str
):
    db: Session = session_local()

    try:

        return (
            db.query(Message)
            .filter(
                Message.thread_id == thread_id
            )
            .order_by(
                Message.created_at
            )
            .all()
        )

    finally:

        db.close()


def get_conversations():

    db: Session = session_local()

    try:

        return (
            db.query(Conversation)
            .order_by(
                Conversation.updated_at.desc()
            )
            .all()
        )

    finally:

        db.close()



def update_conversation_timestamp(
    thread_id: str
):
    db: Session = session_local()

    try:

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.thread_id == thread_id
            )
            .first()
        )

        if conversation:

            conversation.updated_at = datetime.now()

            db.commit()

    finally:

        db.close()