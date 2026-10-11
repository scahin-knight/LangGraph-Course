from agentic_chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import streamlit as st
import uuid


# Generate a unique thread ID
def generate_thread_id():
    return str(uuid.uuid4())


# Add a thread to the conversation list
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Create a new conversation
def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


# Load a previous conversation
def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return state.values.get("messages", [])


# Stream and extract readable AI response text
def generate_response(user_input, config):
    for message_chunk, metadata in chatbot.stream(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=config,
        stream_mode="messages"
    ):
        # Ignore non-AI messages
        if message_chunk.__class__.__name__ not in (
            "AIMessage",
            "AIMessageChunk"
        ):
            continue

        # Ignore chunks generated for tool calls
        if getattr(message_chunk, "tool_calls", None):
            continue

        content = message_chunk.content

        # Handle plain text responses
        if isinstance(content, str):
            if content:
                yield content

        # Handle list-based content blocks
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            yield text


# Application title
st.title("Agentic Chatbot with LangGraph")


# Initialize session state
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []


# Register current conversation
add_thread(st.session_state["thread_id"])


# ========================= Sidebar =========================

st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()


# Display saved conversations
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(
        thread_id,
        key=f"thread_{thread_id}"
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

            content = message.content

            # Convert list-based content to readable text
            if isinstance(content, list):
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                ]
                content = "".join(text_parts)

            if isinstance(content, str) and content:
                temp_messages.append({
                    "role": role,
                    "content": content
                })

        st.session_state["message_history"] = temp_messages
        st.rerun()


# ========================= Main Chat Interface =========================

# Display conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
user_input = st.chat_input("Type here")

if user_input:
    # Save and display the user message
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Current conversation configuration
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    # Stream the assistant response
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            generate_response(user_input, CONFIG)
        )

    # Save the completed assistant response
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })