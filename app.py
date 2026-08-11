import streamlit as st
import threading
import time

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
)

import uuid

# from backend import chatbot

from observability import tracing

# from core.llm import title_llm

from db.repository import (
    create_conversation,
    conversation_exists,
    save_message,
    get_conversations,
    get_messages,
    update_conversation_timestamp,
    update_conversation_title,
)

from ui.sidebar import render_sidebar
from ui.messages import render_messages
from ui.documents import render_documents
from ui.uploader import render_uploader

# from services.document_service import DocumentService

# from rag.embedding.embedding_model import (
#     get_embedding_model,
# )

# from rag.retrieval.reranker import (
#     get_reranker,
# )

from utils.logger import get_logger

from concurrent.futures import ThreadPoolExecutor


# ==========================================================
# Startup Timing
# ==========================================================

APP_START = time.perf_counter()

logger = get_logger(__name__)

logger.info(
    "[STARTUP] app.py execution started."
)


# ==========================================================
# Chatbot
# ==========================================================

_chatbot = None
_chatbot_lock = threading.Lock()


def get_chatbot():

    global _chatbot

    if _chatbot is not None:

        logger.info(
            "[STARTUP] Returning cached chatbot instance."
        )

        return _chatbot

    with _chatbot_lock:

        if _chatbot is None:

            start = time.perf_counter()

            logger.info(
                "[STARTUP] Initializing chatbot backend..."
            )

            from backend import chatbot

            _chatbot = chatbot

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "[STARTUP] Chatbot backend initialized "
                "in %.3f sec.",
                elapsed,
            )

            logger.info(
                "[STARTUP] Total time since app start: "
                "%.3f sec.",
                time.perf_counter() - APP_START,
            )

    return _chatbot


# ==========================================================
# Services
# ==========================================================

@st.cache_resource
def get_document_service():

    start = time.perf_counter()

    logger.info(
        "[STARTUP] Initializing DocumentService..."
    )

    from services.document_service import (
        DocumentService,
    )

    service = DocumentService()

    elapsed = (
        time.perf_counter()
        - start
    )

    logger.info(
        "[STARTUP] DocumentService initialized "
        "in %.3f sec.",
        elapsed,
    )

    logger.info(
        "[STARTUP] Total time since app start: "
        "%.3f sec.",
        time.perf_counter() - APP_START,
    )

    return service


# ==========================================================
# Background Model Warm-up
# ==========================================================

@st.cache_resource
def start_model_warmup():

    def warm_embedding():
        from rag.embedding.embedding_model import (
            get_embedding_model,
        )
        try:

            logger.info(
                "Starting embedding model warm-up."
            )

            embedder = get_embedding_model()

            embedder.embed_query(
                "warmup query"
            )

            logger.info(
                "Embedding model warmed successfully."
            )

        except Exception:

            logger.exception(
                "Embedding model warm-up failed."
            )

    def warm_reranker():

        try:

            logger.info(
                "Starting reranker warm-up."
            )

            from rag.services.retrieval_service import (
                get_retrieval_service,
            )

            retrieval_service = (
                get_retrieval_service()
            )

            retrieval_service.reranker.warm_up()

            logger.info(
                "Reranker warmed successfully."
            )

        except Exception:

            logger.exception(
                "Reranker warm-up failed."
            )

    def warm_models():

        logger.info(
            "Starting parallel background model warm-up."
        )

        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="model-warmup",
        ) as executor:

            embedding_future = executor.submit(
                warm_embedding
            )

            reranker_future = executor.submit(
                warm_reranker
            )

            embedding_future.result()
            reranker_future.result()

        logger.info(
            "Parallel background model warm-up completed."
        )

    thread = threading.Thread(
        target=warm_models,
        daemon=True,
        name="rag-model-warmup",
    )

    thread.start()


# ==========================================================
# Start background warm-up
# ==========================================================

logger.info(
    "[STARTUP] Starting background model warm-up at %.3f sec.",
    time.perf_counter() - APP_START,
)

start_model_warmup()

logger.info(
    "[STARTUP] Background warm-up launched at %.3f sec.",
    time.perf_counter() - APP_START,
)


# ==========================================================
# Get services
# ==========================================================

logger.info(
    "[STARTUP] Skipping DocumentService initialization during startup."
)

logger.info(
    "[STARTUP] Top-level application setup completed at %.3f sec.",
    time.perf_counter() - APP_START,
)

########Utility Functions###############3

def get_thread_id():
    thread_id=str(uuid.uuid4())
    return thread_id

def reset_chat():
    thread_id=get_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"]=[]
    


def generate_chat_title(user_query):

    from core.llm import title_llm

    prompt = f"""
Generate a concise title for this conversation.

Rules:
- Maximum 5 words
- No quotes
- No punctuation
- Return title only

Query:
{user_query}
"""

    response = title_llm.invoke(prompt)

    return response.content.strip()

###################### ChatArea Design #############################3
st.markdown("""
<style>
.user-msg {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0;
}

.user-bubble {
    background-color: #DCF8C6;
    color: black;
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 70%;
}

.ai-msg {

    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}

.ai-bubble {
    background-color: #F1F0F0;
    color: black;
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)
######################Session Setup#########################

if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]

if "thread_id" not in st.session_state:
    st.session_state["thread_id"]=get_thread_id()

if "attached_documents" not in st.session_state:
    st.session_state["attached_documents"] = []

if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = set()

################ Sidebar ###################

conversations = get_conversations()

new_chat, selected_thread = render_sidebar(
    conversations
)
if new_chat:

    st.session_state["thread_id"] = str(uuid.uuid4())

    st.session_state["message_history"] = []

    st.session_state["attached_documents"] = []
    st.session_state["processed_uploads"] = set()
    st.rerun()

# ---------------------------------------------------------
# Load Selected Conversation
# ---------------------------------------------------------

if selected_thread:

    db_messages = get_messages(selected_thread)

    st.session_state["thread_id"] = selected_thread

    st.session_state["message_history"] = []
    st.session_state["processed_uploads"] = set()
    for message in db_messages:

        st.session_state["message_history"].append(

            {
                "role": message.role,
                "content": message.content,
            }

        )
    document_service = get_document_service()
    st.session_state["attached_documents"] = (

        document_service.list_documents(
            selected_thread
        )

    )

    st.rerun()
################### Main UI ###################

logger.info(
    "[STARTUP] UI rendering begins at %.3f sec.",
    time.perf_counter() - APP_START,
)

render_messages(
    st.session_state["message_history"]
)

logger.info(
    "[STARTUP] Messages rendered at %.3f sec.",
    time.perf_counter() - APP_START,
)
delete_document = render_documents(

    st.session_state[
        "attached_documents"
    ]

)

if delete_document:
    document_service = get_document_service()
    document = next(

        doc

        for doc in st.session_state[
            "attached_documents"
        ]

        if doc.id == delete_document

    )

    document_service.remove_document(
        document
    )

    st.session_state[
        "attached_documents"
    ] = document_service.list_documents(

        st.session_state[
            "thread_id"
        ]

    )

    st.rerun()

uploaded_files = render_uploader(st.session_state["thread_id"])

if uploaded_files:

    new_files = []

    for uploaded_file in uploaded_files:

        # Skip files we've already processed during this session
        if uploaded_file.name in st.session_state["processed_uploads"]:
            continue

        st.session_state["processed_uploads"].add(
            uploaded_file.name
        )

        new_files.append(uploaded_file)

    if new_files:
        document_service = get_document_service()
        thread_id = st.session_state["thread_id"]

        if not conversation_exists(thread_id):

            create_conversation(
                thread_id=thread_id,
                title="New Chat",
            )

        with st.status(
            "Processing documents...",
            expanded=True,
        ) as status:

            st.write("📄 Processing documents...")

            st.session_state["attached_documents"] = (
                document_service.upload_documents(
                    thread_id=thread_id,
                    uploaded_files=new_files,
                )
            )

            status.update(
                label="✅ Documents indexed successfully",
                state="complete",
                expanded=False,
            )

        st.rerun()



logger.info(
    "[STARTUP] Chat input reached / UI ready at %.3f sec.",
    time.perf_counter() - APP_START,
)

user_input=st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append(
        {"role":"user",
         "content":user_input
         }
    )
    if len(st.session_state["message_history"]) > 1:
        save_message(
            thread_id=st.session_state["thread_id"],
            role="user",
            content=user_input
        )

    with st.chat_message("user"):
        st.text(user_input)

        if len(st.session_state["message_history"]) == 1:

            title = generate_chat_title(user_input)

            if not conversation_exists(
                st.session_state["thread_id"]
            ):

                create_conversation(

                    thread_id=st.session_state["thread_id"],

                    title=title,
                )

            else:

                update_conversation_title(

                    thread_id=st.session_state["thread_id"],

                    title=title,
                )

        save_message(
            thread_id=st.session_state["thread_id"],
            role="user",
            content=user_input
        )
    # response=chatbot.invoke({"messages":[HumanMessage(content=user_input)]},config=config)

    config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }
    ########### STREAMING  ###################

    with st.chat_message("assistant"):
        chatbot = get_chatbot()
        def response_stream():

            streamed_text = ""

            for chunk, metadata in chatbot.stream(

                    {
                        "messages": [HumanMessage(content=user_input)],
                        "query": user_input,
                        "route": "",
                        "tool_call": None,
                        "tool_result": None,
                        "final_answer": None,
                        "tool_history": [],
                        "tool_results": [],
                        "observation": None,
                        "iteration": 0,
                        "status": "PLANNING",
                        "error": None,
                        "metadata": {
                            "thread_id": st.session_state["thread_id"],
                             "documents": [
                                    doc.filename
                                    for doc in st.session_state["attached_documents"]
                            ],

                        },
                    },

                    config=config,

                    stream_mode="messages",

            ):

                # Stream only the final answer
                node=metadata.get("langgraph_node")
                if(node) not in ("general","answer"):
                    continue

                if not isinstance(chunk, AIMessageChunk):
                    continue

                text = chunk.content or ""

                if not text:
                    continue

                if text.startswith(streamed_text):
                    new_text = text[len(streamed_text):]
                else:
                    new_text = text

                streamed_text += new_text

                yield new_text


        ai_message = st.write_stream(response_stream())

        state = chatbot.get_state(config)

        citations = state.values.get(
            "citations",
            [],
        )
        if citations:

            with st.expander(
                    "📄 Sources"
            ):

                for citation in citations:
                    st.write(
                        f"**{citation['source']}** (Page {citation['page']})"
                    )

    st.session_state["message_history"].append(
        {"role": "assistant",
         "content": ai_message
         }
    )
    save_message(
        thread_id=st.session_state["thread_id"],
        role="assistant",
        content=ai_message
    )
    update_conversation_timestamp(
        st.session_state["thread_id"]
    )



