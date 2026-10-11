
import streamlit as st
from langchain_core.messages import HumanMessage
from agentic_chatbot_backend import chatbot

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖"
)

st.title("Agentic Chatbot with LangGraph")

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    def generate_response():
        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            content = message_chunk.content

            if isinstance(content, str) and content:
                yield content

            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        yield block
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            yield text

    with st.chat_message("assistant"):
        ai_message = st.write_stream(generate_response())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )