
import streamlit as st
import uuid

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

from agentic_chatbot_db_backend import chatbot, get_all_threads


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []

    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = chatbot.get_state(config=config)

    return state.values.get("messages", [])


def extract_text(content):
    """Extract readable text from string or structured content."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")

                    if isinstance(text, str):
                        text_parts.append(text)

        return "".join(text_parts)

    return ""


def stream_response(user_input, config):
    """Stream only readable assistant text from LangGraph."""
    for message_chunk, metadata in chatbot.stream(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=config,
        stream_mode="messages",
    ):
        # Ignore user messages, tool messages and other message types.
        if not isinstance(message_chunk, AIMessage):
            continue

        # Ignore chunks that contain no readable text.
        text = extract_text(message_chunk.content)

        if text:
            yield text


# --------------------------------------------------
# Initialize session state
# --------------------------------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

add_thread(st.session_state["thread_id"])


# --------------------------------------------------
# Main page
# --------------------------------------------------

st.title("Agentic Chatbot with LangGraph")


# --------------------------------------------------
# Sidebar: conversation history
# --------------------------------------------------

st.sidebar.title("My Conversations")

if st.sidebar.button("＋ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()


for thread_id in reversed(st.session_state["chat_threads"]):
    if st.sidebar.button(
        f"Chat {thread_id[:8]}",
        key=f"thread_{thread_id}",
        use_container_width=True,
    ):
        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)
        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"

            elif isinstance(message, AIMessage):
                role = "assistant"

            else:
                continue

            content = extract_text(message.content)

            if content:
                temp_messages.append({
                    "role": role,
                    "content": content,
                })

        st.session_state["message_history"] = temp_messages
        st.rerun()


# --------------------------------------------------
# Display current conversation
# --------------------------------------------------

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat input and streaming response
# --------------------------------------------------

user_input = st.chat_input("Type your message...")

if user_input:
    # Add and display the user message
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Use the current conversation thread for checkpointing
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"],
        },
        "run_name": "chat_trace",
    }

    # Stream the assistant response
    with st.chat_message("assistant"):
        try:
            ai_message = st.write_stream(
                stream_response(user_input, CONFIG)
            )

            # Save the completed response in the UI history
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": ai_message,
            })

        except Exception as e:
            st.error(f"An error occurred: {e}")