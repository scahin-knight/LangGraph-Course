
from agentic_chatbot_hitl_backend import (
    chatbot,
    get_all_threads,
    ingest_rag_document,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)

from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_thread_id():
    return str(uuid.uuid4())


def extract_text(content):
    """Extract readable text from Gemini/LangChain message content."""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if content.get("type") in ("text", "output_text"):
            text = content.get("text", "")
            return text if isinstance(text, str) else ""

        return ""

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                if block.get("type") in ("text", "output_text"):
                    text = block.get("text", "")
                    if isinstance(text, str) and text:
                        text_parts.append(text)

        return "".join(text_parts)

    return ""


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["pending_hitl"] = None
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = chatbot.get_state(config)
    return state.values.get("messages", [])


# ============================================================
# HUMAN-IN-THE-LOOP HELPERS
# ============================================================

def get_pending_interrupt(thread_id):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        state = chatbot.get_state(config)

        interrupts = getattr(state, "interrupts", ()) or ()

        if interrupts:
            return interrupts[0]

        for task in getattr(state, "tasks", ()) or ():
            task_interrupts = getattr(task, "interrupts", ()) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value),
    }


def sync_pending_interrupt(thread_id):
    pending = get_pending_interrupt(thread_id)

    if pending is not None:
        save_pending_interrupt(thread_id, pending)

    else:
        current = st.session_state.get("pending_hitl")

        if current and current.get("thread_id") == thread_id:
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    pending = st.session_state.get("pending_hitl")

    if not pending:
        st.warning("There is no pending action to approve or reject.")
        return

    thread_id = pending["thread_id"]

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        response_parts = []

        with st.chat_message("assistant"):
            status_box = st.status(
                "Resuming the requested action...",
                expanded=True,
            )

            def resumed_stream():
                for chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=config,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, ToolMessage):
                        status_box.update(
                            label=f"Using tool: {chunk.name}",
                            state="running",
                        )
                        continue

                    if isinstance(chunk, (AIMessage, AIMessageChunk)):
                        # Do not display tool-call arguments or metadata.
                        if getattr(chunk, "tool_calls", None):
                            continue

                        text = extract_text(chunk.content)

                        if text:
                            response_parts.append(text)
                            yield text

            st.write_stream(resumed_stream())

            next_interrupt = get_pending_interrupt(thread_id)

            if next_interrupt is not None:
                save_pending_interrupt(thread_id, next_interrupt)
                status_box.update(
                    label="Another approval is required",
                    state="complete",
                    expanded=False,
                )
            else:
                st.session_state["pending_hitl"] = None
                status_box.update(
                    label="Action completed",
                    state="complete",
                    expanded=False,
                )

        complete_response = "".join(response_parts).strip()

        if complete_response:
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": complete_response,
            })

        st.rerun()

    except Exception as error:
        st.error(f"Could not resume the action: {error}")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("Agentic Chatbot with LangGraph")


# ============================================================
# SESSION STATE
# ============================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

add_thread(st.session_state["thread_id"])

sync_pending_interrupt(st.session_state["thread_id"])


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

for thread_id in st.session_state["chat_threads"][::-1]:
    label = f"Chat {thread_id[:8]}"

    if st.sidebar.button(label, key=f"thread_{thread_id}"):
        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)
        history = []

        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({
                    "role": "user",
                    "content": extract_text(message.content),
                })

            elif isinstance(message, AIMessage):
                # Hide assistant messages that only request tool execution.
                if getattr(message, "tool_calls", None):
                    continue

                text = extract_text(message.content)

                if text:
                    history.append({
                        "role": "assistant",
                        "content": text,
                    })

        st.session_state["message_history"] = history
        sync_pending_interrupt(thread_id)
        st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(extract_text(message["content"]))


# ============================================================
# HUMAN APPROVAL INTERFACE
# ============================================================

pending = st.session_state.get("pending_hitl")

current_thread_has_pending_hitl = (
    pending is not None
    and pending.get("thread_id") == st.session_state["thread_id"]
)

if current_thread_has_pending_hitl:
    st.warning(
        "Human approval required\n\n"
        + pending["prompt"]
    )

    approve_col, reject_col = st.columns(2)

    with approve_col:
        if st.button(
            "✅ Approve Purchase",
            type="primary",
            use_container_width=True,
        ):
            resume_hitl_execution("yes")

    with reject_col:
        if st.button(
            "❌ Reject Purchase",
            use_container_width=True,
        ):
            resume_hitl_execution("no")


# ============================================================
# CHAT INPUT WITH PDF UPLOAD
# ============================================================

submission = st.chat_input(
    "Ask me anything...",
    accept_file=True,
    file_type=["pdf"],
    disabled=current_thread_has_pending_hitl,
)

if submission:
    user_input = (submission.text or "").strip()
    uploaded_files = submission.files or []

    # --------------------------------------------------------
    # PROCESS PDF
    # --------------------------------------------------------

    pdf_processed = False

    for uploaded_pdf in uploaded_files:
        temporary_file_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temporary_file:
                temporary_file.write(uploaded_pdf.getvalue())
                temporary_file_path = temporary_file.name

            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                ingest_rag_document(temporary_file_path)

            st.success(f"{uploaded_pdf.name} processed successfully.")
            pdf_processed = True

        except Exception as error:
            st.error(f"PDF processing failed: {error}")

        finally:
            if (
                temporary_file_path
                and os.path.exists(temporary_file_path)
            ):
                os.remove(temporary_file_path)

    # If the user only uploaded a PDF, do not send an empty query.
    if not user_input and pdf_processed:
        user_input = (
            "Summarize the uploaded PDF and explain its main points."
        )

    # --------------------------------------------------------
    # RUN CHATBOT
    # --------------------------------------------------------

    if user_input:
        st.session_state["message_history"].append({
            "role": "user",
            "content": user_input,
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        config = {
            "configurable": {
                "thread_id": st.session_state["thread_id"],
            },
            "metadata": {
                "thread_id": st.session_state["thread_id"],
            },
            "run_name": "chat_trace",
        }

        response_parts = []

        with st.chat_message("assistant"):
            status_box = None

            def assistant_stream():
                nonlocal_status = {"box": None}

                for chunk, metadata in chatbot.stream(
                    {
                        "messages": [
                            HumanMessage(content=user_input)
                        ]
                    },
                    config=config,
                    stream_mode="messages",
                ):
                    # Tool results are not part of the visible answer.
                    if isinstance(chunk, ToolMessage):
                        tool_name = getattr(chunk, "name", "tool")

                        if nonlocal_status["box"] is None:
                            nonlocal_status["box"] = st.status(
                                f"🔧 Using {tool_name}...",
                                expanded=True,
                            )
                        else:
                            nonlocal_status["box"].update(
                                label=f"🔧 Using {tool_name}...",
                                state="running",
                            )

                        continue

                    # Only stream assistant-generated text.
                    if isinstance(chunk, (AIMessage, AIMessageChunk)):
                        # Ignore tool-call chunks and their arguments.
                        if getattr(chunk, "tool_calls", None):
                            continue

                        text = extract_text(chunk.content)

                        if text:
                            response_parts.append(text)
                            yield text

                # Check whether the graph paused for human approval.
                pending_interrupt = get_pending_interrupt(
                    st.session_state["thread_id"]
                )

                if pending_interrupt is not None:
                    save_pending_interrupt(
                        st.session_state["thread_id"],
                        pending_interrupt,
                    )

                    yield (
                        "\n\n⚠️ This action requires your approval. "
                        "Use the approval buttons below."
                    )

                if nonlocal_status["box"] is not None:
                    if get_pending_interrupt(
                        st.session_state["thread_id"]
                    ) is not None:
                        nonlocal_status["box"].update(
                            label="⏸️ Waiting for approval",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        nonlocal_status["box"].update(
                            label="✅ Tool finished",
                            state="complete",
                            expanded=False,
                        )

            st.write_stream(assistant_stream())

        # Save only the actual readable assistant text.
        complete_response = "".join(response_parts).strip()

        if complete_response:
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": complete_response,
            })

        # Show approval buttons immediately if an interrupt occurred.
        if (
            st.session_state.get("pending_hitl") is not None
            and st.session_state["pending_hitl"].get("thread_id")
            == st.session_state["thread_id"]
        ):
            st.rerun()