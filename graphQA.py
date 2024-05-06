import streamlit as st
from GraphRAG import NLQ, GraphEmbeddings
from RAG import RAG
import os


nlq = NLQ()
graph_embeddings = GraphEmbeddings()
rag = RAG()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

conversation_file_path = "conversation.txt"

# Create a sidebar for navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["QA with RAG", "QA with Graph RAG", "Knowledge Graph",
                                           "Recommendations", "Semantic Search"])

# QA with RAG Page
if selected_page == "QA with RAG":
    st.title("Ask me about Food and Disease!")

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
        response = rag.get_response(prompt)

        # Display model response in chat message container
        with st.chat_message("bot"):
            st.markdown(response)

        # Add model response to chat history
        st.session_state.messages.append({"role": "bot", "content": response})

        with open(conversation_file_path, "a") as file:
            file.write(f"user: {prompt}\n")
            file.write(f"bot: {response}\n\n")

# QA with Graph RAG Page
if selected_page == "QA with Graph RAG":
    st.title("Ask me about Food and Disease!")

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
        # response = nlq.get_response(prompt)
        response = graph_embeddings.get_response(prompt)

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

# Recommendations Page
if selected_page == "Recommendations":
    st.title("Recommended Food for your Disease!")

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
        response = nlq.get_response(prompt)

        # Display model response in chat message container
        with st.chat_message("bot"):
            st.markdown(response)

        # Add model response to chat history
        st.session_state.messages.append({"role": "bot", "content": response})

        with open(conversation_file_path, "a") as file:
            file.write(f"user: {prompt}\n")
            file.write(f"bot: {response}\n\n")

# About Page
elif selected_page == "Semantic Search":
    st.title("About")
    st.write("Semantic Search Page")