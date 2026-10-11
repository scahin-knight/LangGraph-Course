
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()

# Initialize Gemini with streaming enabled
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.7,
    streaming=True
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]

    # Generate the assistant response
    response = llm.invoke(messages)

    return {"messages": [response]}


# SQLite connection for persistent conversation memory
conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpoint = SqliteSaver(conn)

# Build the LangGraph workflow
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile with SQLite checkpointing
chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads():
    all_threads = set()

    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config["configurable"].get("thread_id")

        if thread_id:
            all_threads.add(thread_id)

    return list(all_threads)