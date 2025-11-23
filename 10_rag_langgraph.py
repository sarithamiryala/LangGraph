from langchain_google_genai import ChatGoogleGenerativeAI 
from langgraph.graph import START,StateGraph,END 
import os 
from dotenv import load_dotenv 
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.vectorstores import FAISS 
from typing import TypedDict, Annotated 
from langgraph.graph.message import add_messages 
from langchain_core.messages import HumanMessage, BaseMessage 
from langgraph.prebuilt import ToolNode, tools_condition 
from langchain_core.tools import tool



load_dotenv() 
os.environ['GOOGLE_API_KEY'] = os.getenv('gemini_api_key')
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  


## Loading the pdf 
loader = PyPDFLoader("Miryala_et_al-2025-Discover_Artificial_Intelligence.pdf") 
docs =loader.load() 

## Text Splitter 
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
chunks = splitter.split_documents(docs) 

## Enbedding and storing in vector store 
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)


retriever = vector_store.as_retriever(search_type ='similarity', search_kwargs = {'k': 4})  


@tool
def rag_tool(query):
    """
    Retrieve relevant information from the pdf documents.
    Use this tool when asks factual / Conceptual questions 
    that might be answered from the stored documents. """ 

    result = retriever.invoke(query) 
  

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result] 

    return {
        "query" : query,
        "context": context,
        "metadata": metadata
    } 

# query = "who ar the authors in this paper?"
# result = rag_tool(query) 
# print(result) 

tools = [rag_tool] 
llm_with_tools = llm.bind_tools(tools) 

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_node(state: ChatState):

    messages= state["messages"]
    response = llm_with_tools.invoke(messages) 

    return {"messages": [response]} 

tool_node = ToolNode(tools) 

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node) 
graph.add_node('tools', tool_node) 

graph.add_edge(START, 'chat_node') 
graph.add_conditional_edges('chat_node', tools_condition) 

graph.add_edge('tools', 'chat_node')
chatbot = graph.compile() 

result = chatbot.invoke(
    {
        "messages" : [
            HumanMessage(content=("What is the main objective of this paper and which methos is being used and what is the accuracy achieved "))
        ]
    } 
)

print(result['messages'][-1].content)