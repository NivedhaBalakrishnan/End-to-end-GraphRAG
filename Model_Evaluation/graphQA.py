import streamlit as st
from graphRAG import GraphRAG
import os

model = GraphRAG()
get_response = model.get_response

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

conversation_file_path = "conversation.txt"

# Create a sidebar for navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["Chat", "Knowledge Graph", "About"])

# Chat Page
if selected_page == "Chat":
    st.title("Ask me about Inflammation!")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get model response
        response = get_response(prompt)

        # Display model response in chat message container
        with st.chat_message("bot"):
            st.markdown(response)

        # Add model response to chat history
        st.session_state.messages.append({"role": "bot", "content": response})

        with open(conversation_file_path, "a") as file:
            file.write(f"user: {prompt}\n")
            file.write(f"bot: {response}\n\n")

# Knowledge Graph Page
elif selected_page == "Knowledge Graph":
    st.title("Knowledge Graph")
    # Add your code to display the knowledge graph here
    st.write("This is the knowledge graph page.")

# About Page
elif selected_page == "About":
    st.title("About")
    st.write("This is a chatbot that can answer questions about inflammation. It is based on the [GraphRAG]")