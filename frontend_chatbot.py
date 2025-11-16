import streamlit as st
from backend_chatbot import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

st.title("💬 AI Chatbot")

# Display conversation history
for message in st.session_state['message_history']:
    if message['role'] == 'user':
        st.markdown(f"**👤 You:** {message['content']}")
    else:
        st.markdown(f"**🤖 Assistant:** {message['content']}")

# User input box
user_input = st.text_input('Type your message here:')

if st.button("Send") and user_input:
    # Store user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    st.markdown(f"**👤 You:** {user_input}")

    # Get bot response
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    ai_message = response['messages'][-1].content

    # Store bot message
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    st.markdown(f"**🤖 Assistant:** {ai_message}")
