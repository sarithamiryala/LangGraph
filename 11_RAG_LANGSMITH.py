from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph, END
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated
from langsmith import traceable


# Load env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("gemini_api_key")

# LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "rag-gemini-langgraph"

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# Load PDF
loader = PyPDFLoader("Miryala_et_al-2025-Discover_Artificial_Intelligence.pdf")
docs = loader.load()

# Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embeddings + FAISS Vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})


@traceable(run_type="tool", name="RAG Retrieval Tool")
@tool
def rag_tool(query: str):
    """Retrieve relevant information from the stored PDF."""
    docs = retriever.invoke(query)
    context = [d.page_content for d in docs]
    metadata = [d.metadata for d in docs]
    return {"query": query, "context": context, "metadata": metadata}


tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@traceable(run_type="llm", name="Chat Node")
def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)

# LangGraph workflow
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# Compile graph (tracing happens automatically via env variables)
chatbot = graph.compile()

# Ask question
result = chatbot.invoke({
    "messages": [
        HumanMessage(content="Give the Abstract of the paper")
    ]
})

print(result["messages"][-1].content)
