import streamlit as st
from deprecated.graphRAG import GraphRAG
from RAG import RAG
import os

model = GraphRAG()
get_response = model.get_response

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
        response = get_response(prompt)

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

# import os
# import logging
# import streamlit as st
# from graphRAG import GraphRAG
# from neo4j_append import Neo4jAppend
# from helper import Helper


# # Configure logging
# log_file = "app.log"
# log_format = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)
# logger = logging.getLogger(__name__)

# # # Set up logging
# # hlp = Helper()
# # # # hlp.clear_log_file()
# # log = hlp.get_logger()
# # log.info("Starting the app")


# model = GraphRAG()
# get_response = model.get_response

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# conversation_file_path = "conversation.txt"

# selected_page = st.sidebar.radio("Methods", ["Graph RAG with Documents",
#                                               "Graph RAG with ER", ])


# # Chat Page
# if selected_page == "Graph RAG with Documents":
#     st.title("Ask me about Inflammation!")

#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Accept user input
#     if prompt := st.chat_input("What is up?"):
#         # Display user message in chat message container
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         # Get model response
#         response, _, _ = get_response(prompt)
#         # response = ""

#         # Display model response in chat message container
#         with st.chat_message("bot"):
#             st.markdown(response)

#         # Add model response to chat history
#         st.session_state.messages.append({"role": "bot", "content": response})

#         with open(conversation_file_path, "a") as file:
#             file.write(f"user: {prompt}\n")
#             file.write(f"bot: {response}\n\n")


# # Knowledge Graph Page
# elif selected_page == "Graph RAG with ER":
#     st.title("Ask me about Inflammation!")
#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Accept user input
#     if prompt := st.chat_input("What is up?"):
#         # Display user message in chat message container
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         # Get model response
#         response, _, _ = get_response(prompt)


#         # Display model response in chat message container
#         with st.chat_message("bot"):
#             st.markdown(response)

#         # Add model response to chat history
#         st.session_state.messages.append({"role": "bot", "content": response})

#         with open(conversation_file_path, "a") as file:
#             file.write(f"user: {prompt}\n")
#             file.write(f"bot: {response}\n\n")



# def main():
#     print("Starting the app")
#     st.sidebar.header("Upload your document here")
#     uploaded_file = st.sidebar.file_uploader("", type=["pdf", "txt"], label_visibility="hidden")
#     if st.sidebar.button("Submit"):
#         if uploaded_file is not None:
#             neo4j_append = Neo4jAppend()
#             was_success = neo4j_append.process_file(uploaded_file)
#             if was_success:
#                 st.sidebar.success("Successfully added to Neo4j!")
#             else:
#                 st.sidebar.error("Error adding to Neo4j!")


# if __name__ == "__main__":
#     main()