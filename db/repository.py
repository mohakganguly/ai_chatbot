from sqlalchemy.orm import Session

from db.database import session_local
from db.models import (
    Conversation,
    Message,
    ConversationDocument,
)

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

def save_document(
    thread_id: str,
    filename: str,
    filepath: str,
):
    db: Session = session_local()

    try:

        document = ConversationDocument(

            id=str(uuid.uuid4()),

            thread_id=thread_id,

            filename=filename,

            filepath=filepath,

            status="INDEXED",
        )

        db.add(document)

        db.commit()

        return document

    finally:

        db.close()

def get_documents(
    thread_id: str,
):
    db: Session = session_local()

    try:

        return (

            db.query(
                ConversationDocument
            )

            .filter(
                ConversationDocument.thread_id == thread_id
            )

            .order_by(
                ConversationDocument.created_at
            )

            .all()

        )

    finally:

        db.close()

def delete_document(
    document_id: str,
):
    db: Session = session_local()

    try:

        document = (

            db.query(
                ConversationDocument
            )

            .filter(
                ConversationDocument.id == document_id
            )

            .first()

        )

        if document:

            db.delete(document)

            db.commit()

    finally:

        db.close()

def delete_documents(
    thread_id: str,
):
    db: Session = session_local()

    try:

        (

            db.query(
                ConversationDocument
            )

            .filter(
                ConversationDocument.thread_id == thread_id
            )

            .delete()

        )

        db.commit()

    finally:

        db.close()

def update_document_status(
    document_id: str,
    status: str,
):
    db: Session = session_local()

    try:

        document = (

            db.query(
                ConversationDocument
            )

            .filter(
                ConversationDocument.id == document_id
            )

            .first()

        )

        if document:

            document.status = status

            db.commit()

    finally:

        db.close()


def get_document(
    document_id: str,
):
    db: Session = session_local()

    try:

        return (

            db.query(
                ConversationDocument
            )

            .filter(
                ConversationDocument.id == document_id
            )

            .first()

        )

    finally:

        db.close()

def conversation_exists(
    thread_id: str,
) -> bool:

    db: Session = session_local()

    try:

        return (

            db.query(Conversation)

            .filter(
                Conversation.thread_id == thread_id
            )

            .first()

            is not None
        )

    finally:

        db.close()

def update_conversation_title(
    thread_id: str,
    title: str,
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

            conversation.title = title

            conversation.updated_at = datetime.now()

            db.commit()

    finally:

        db.close()

def document_exists(
    thread_id: str,
    filename: str,
) -> bool:

    db = session_local()

    try:

        return (

            db.query(ConversationDocument)

            .filter(
                ConversationDocument.thread_id == thread_id,
                ConversationDocument.filename == filename,
            )

            .first()

            is not None
        )

    finally:

        db.close()