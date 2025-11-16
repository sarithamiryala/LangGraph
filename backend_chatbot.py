from langgraph.graph import StateGraph, START,END 
from langchain_google_genai import ChatGoogleGenerativeAI 
import os 
from typing import TypedDict, Annotated, Literal 
from langchain_core.messages import BaseMessage, HumanMessage 
from dotenv import load_dotenv 
from pydantic import BaseModel,Field 
from langgraph.checkpoint.memory import InMemorySaver 

load_dotenv() 

os.environ['GOOGLE_API_KEY'] = os.getenv('gemini_api_key') 
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 

## Defining state
from langgraph.graph.message import add_messages
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages] 

## creating node function 
def chat_node(state: ChatState): 

    messages = state['messages'] 

    response = model.invoke(messages) 

    return { "messages": [response]}   

checkpointer = InMemorySaver()
graph = StateGraph(ChatState) 

#Nodes 
graph.add_node('chat_node', chat_node) 

# Edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END) 

chatbot = graph.compile(checkpointer=checkpointer)

