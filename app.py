import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from datetime import datetime
from backend import chatbot
from observability import tracing
from core.llm import title_llm
######## Repository Functions ########
from db.repository import (
    create_conversation,
    save_message,
    get_conversations,
    get_messages,
    update_conversation_timestamp
)
from services.ingestion_service import IngestionService
from langchain_core.messages import AIMessageChunk
# ==========================================================
# Services
# ==========================================================

ingestion_service = IngestionService()
########Utility Functions###############3

def get_thread_id():
    thread_id=str(uuid.uuid4())
    return thread_id

def reset_chat():
    thread_id=get_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"]=[]
    # add_threads(thread_id)

# def add_threads(
#     thread_id,
#     title="New Chat",
#     first_message=""
# ):
#     if thread_id not in st.session_state["chat_threads"]:
#
#         st.session_state["chat_threads"][thread_id] = {
#             "title": title,
#             "created_at": datetime.now(),
#             "updated_at": datetime.now(),
#             "first_message": first_message
#         }

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

def generate_chat_title(user_query):

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

if "current_documents" not in st.session_state:
    st.session_state["current_documents"] = []

# if "chat_threads" not in st.session_state:
#     st.session_state["chat_threads"]={}

# add_threads(st.session_state["thread_id"])

################ Sidebar ###################

st.sidebar.title("AI Chatbot")

# ==========================================================
# Knowledge Base
# ==========================================================

st.sidebar.header("DOCUMENTS")

uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt", "md", "markdown", "html", "htm"],
    accept_multiple_files=True,
)

if uploaded_files:

    st.sidebar.success(
        f"{len(uploaded_files)} file(s) selected."
    )

    if st.sidebar.button(
        "📥 Index Documents",
        use_container_width=True,
    ):

        with st.spinner(
            "Indexing documents..."
        ):

            saved_paths = (
                ingestion_service.save_uploaded_files(
                    uploaded_files
                )
            )
            result = (
                ingestion_service.ingest_documents(
                    saved_paths
                )
            )

            # Store currently uploaded documents
            st.session_state["current_documents"] = [

                file.name

                for file in uploaded_files

            ]



        st.sidebar.success(
            f"""
Indexed Successfully

Documents : {result['documents']}
Chunks : {result['chunks']}
"""
        )

st.sidebar.divider()

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Recents")
# ==========================================================
# Recent Conversations
# ==========================================================

conversations = get_conversations()

for conversation in conversations:

    if st.sidebar.button(
        conversation.title,
        key=conversation.thread_id,
        use_container_width=True,
    ):

        st.session_state["thread_id"] = conversation.thread_id

        db_messages = get_messages(
            conversation.thread_id
        )

        temp_messages = []

        for msg in db_messages:

            temp_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        st.session_state["message_history"] = temp_messages

        st.rerun()
################### Main UI ###################
for message in st.session_state["message_history"]:

    if message["role"] == "user":
        st.markdown(
            f"""
            <div class="user-msg">
                <div class="user-bubble">
                    {message['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div class="ai-msg">
                <div class="ai-bubble">
                    {message['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
# for message in st.session_state["message_history"]:
#     with st.chat_message(message["role"]):
#         st.text(message["content"])

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

        # add_threads(
        #     thread_id=st.session_state["thread_id"],
        #     title=title,
        #     first_message=user_input
        # )

        create_conversation(
            thread_id=st.session_state["thread_id"],
            title=title
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
                            "current_documents":

                                st.session_state.get(
                                    "current_documents",
                                    [],
                                )

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




