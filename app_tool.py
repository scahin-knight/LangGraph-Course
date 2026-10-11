
from agentic_chatbot_tool_backend import chatbot, get_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import streamlit as st
import uuid


# ========================= Helper Functions =========================

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
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


def extract_text(content):
    """Extract readable text from LangChain content blocks."""

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
                    if isinstance(text, str) and text:
                        text_parts.append(text)

        return "".join(text_parts)

    return ""


def format_message(message):
    if isinstance(message, HumanMessage):
        return {
            "role": "user",
            "content": extract_text(message.content),
        }

    if isinstance(message, AIMessage):
        # Skip AI tool-call messages.
        if getattr(message, "tool_calls", None):
            return None

        content = extract_text(message.content)

        if content.strip():
            return {
                "role": "assistant",
                "content": content,
            }

    # Ignore tool messages and other internal messages.
    return None


# ========================= Initialize Session State =========================

st.title("Agentic Chatbot with LangGraph")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

add_thread(st.session_state["thread_id"])


# ========================= Sidebar =========================

st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()


for thread_id in st.session_state["chat_threads"][::-1]:

    label = (
        "Current Chat"
        if thread_id == st.session_state["thread_id"]
        else f"Chat {thread_id[:8]}"
    )

    if st.sidebar.button(label, key=thread_id):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)
        temp_messages = []

        for message in messages:
            formatted = format_message(message)

            if formatted is not None:
                temp_messages.append(formatted)

        st.session_state["message_history"] = temp_messages
        st.rerun()


# ========================= Main Chat Interface =========================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Type here")


# ========================= Process User Input =========================

if user_input:

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"],
        },
        "run_name": "chat_trace",
    }

    # ========================= Assistant Streaming =========================

    with st.chat_message("assistant"):

        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                # Display tool execution status.
                if isinstance(message_chunk, ToolMessage):
                    tool_name = (
                        getattr(message_chunk, "name", None)
                        or "tool"
                    )

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using {tool_name}...",
                            expanded=True,
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using {tool_name}...",
                            state="running",
                            expanded=True,
                        )

                    continue

                if not isinstance(message_chunk, AIMessage):
                    continue

                # Do not display internal tool calls.
                if getattr(message_chunk, "tool_calls", None):
                    continue

                # Extract only actual text from the AI response.
                text = extract_text(message_chunk.content)

                # Ignore empty chunks and metadata-only chunks.
                if text:
                    yield text

        try:
            ai_message = st.write_stream(ai_only_stream())

            # Normalize the final response to a string.
            if not isinstance(ai_message, str):
                ai_message = str(ai_message) if ai_message else ""

            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False,
                )

            # Store the complete assistant response.
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": ai_message,
            })

        except Exception as error:
            st.error(f"Error generating response: {error}")