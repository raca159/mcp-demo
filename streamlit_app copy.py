import requests
import os
import asyncio
from typing import Dict, Any
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import datetime

# Load environment variables from .env file
load_dotenv()

model = AzureChatOpenAI(azure_deployment=os.environ['AZURE_GPT_DEPLOYMENT'])

async def process_prompt(prompt: str) -> Dict[str, Any]:
    mcp_client = MultiServerMCPClient({
        "arxiv": {
            "url": "http://arxiv-server:8000/sse",  # Use Docker service name
            "transport": "sse",
        },
        "docling": {
            "url": "http://docling-server:8001/sse",  # Use Docker service name
            "transport": "sse",
        }
    })
    async with mcp_client as client:
        agent = create_react_agent(model, client.get_tools())
        response = await agent.ainvoke({"messages": prompt}, debug=False)
        messages = [{i.type: i.content} for i in response['messages'] if i.content!='']
        return {"messages": messages}

# Function to query the research API
def query_research_assistant(prompt):
    try:
        return asyncio.run(process_prompt(prompt))
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None
    
def main():
    # Set up the page
    st.set_page_config(page_title="Research Assistant", layout="wide")
    st.title("Research Assistant")

    # Initialize chat history structure
    if "chats" not in st.session_state:
        st.session_state.chats = {
            1: {"title": "New Chat", "messages": []}
        }
    
    # Track the current active chat ID
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = 1
    
    # Track the next chat ID to assign
    if "next_chat_id" not in st.session_state:
        st.session_state.next_chat_id = 2

    # Sidebar for navigation
    with st.sidebar:
        # New chat button
        if st.button("New Chat", key="new_chat_btn"):
            # Create a new chat
            new_id = st.session_state.next_chat_id
            timestamp = datetime.datetime.now().strftime("%H:%M")
            st.session_state.chats[new_id] = {
                "title": f"Chat {new_id} ({timestamp})",
                "messages": []
            }
            st.session_state.active_chat_id = new_id
            st.session_state.next_chat_id += 1
            st.rerun()
        
        # Show clear chat button for current chat
        if st.button("Clear Current Chat"):
            st.session_state.chats[st.session_state.active_chat_id]["messages"] = []
            st.rerun()
        
        # Display list of available chats
        st.write("### Your Chats")
        
        # Create a list of chat options for the radio buttons
        chat_options = {str(chat_id): chat_data["title"] for chat_id, chat_data in st.session_state.chats.items()}
        
        # Convert active_chat_id to string for the radio button
        selected_chat = str(st.session_state.active_chat_id)
        
        # Use radio buttons for chat selection instead of buttons
        selected_chat = st.radio(
            "Select a chat:",
            options=list(chat_options.keys()),
            format_func=lambda x: chat_options[x],
            label_visibility="collapsed",
            index=list(chat_options.keys()).index(selected_chat)
        )
        
        # Update active chat ID when selection changes
        if int(selected_chat) != st.session_state.active_chat_id:
            st.session_state.active_chat_id = int(selected_chat)
            st.rerun()

    # Get current chat data
    current_chat = st.session_state.chats[st.session_state.active_chat_id]
    current_messages = current_chat["messages"]
    
    # Display all messages in the current chat
    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a research question..."):
        # Add user message to chat history
        current_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process response
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                if len(current_messages) == 1:
                    response = query_research_assistant(prompt)
                else:
                    message_history = '\n'.join([
                        f"{msg['role'].capitalize()}:\n{msg['content']}" 
                        for msg in current_messages
                    ])
                    response = query_research_assistant(message_history)
                
                if response:
                    # Process the response
                    full_response = ""
                    
                    for msg in response.get("messages", []):
                        for msg_type, content in msg.items():
                            if msg_type == "ai":
                                full_response += "\n\n" if full_response else ""
                                full_response += content
                    
                    st.markdown(full_response)
                    
                    # Add to chat history
                    current_messages.append({"role": "assistant", "content": full_response})

    st.caption("Powered by LangChain and MCP")
    
if __name__ == '__main__':
    main()