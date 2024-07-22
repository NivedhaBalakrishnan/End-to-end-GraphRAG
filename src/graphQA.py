import streamlit as st
from graphRAG import GraphRAG
from neo4j_append import Neo4jAppend
from helper import Helper


# Configure logging
log_file = "app.log"
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# # Set up logging
# hlp = Helper()
# # # hlp.clear_log_file()
# log = hlp.get_logger()
# log.info("Starting the app")


model = GraphRAG()
get_response = model.get_response

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "messages_graph_rag" not in st.session_state:
    st.session_state.messages_graph_rag = []
if "messages_recommendations" not in st.session_state:
    st.session_state.messages_recommendations = []

conversation_file_path = "conversation.txt"
conversation_file_path2 = "conversation2.txt"
conversation_file_path3 = "conversation3.txt"

# Create a sidebar for navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["QA with RAG", "QA with Graph RAG", "Knowledge Graph",
                                           "Recommendations"])

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
    for message in st.session_state.messages_graph_rag:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to Graph RAG chat history
        st.session_state.messages_graph_rag.append({"role": "user", "content": prompt})

        # Get model response
        response = graphRAG.get_response(prompt)

        # Display model response in chat message container
        with st.chat_message("bot"):
            st.markdown(response)

        # Add model response to Graph RAG chat history
        st.session_state.messages_graph_rag.append({"role": "bot", "content": response})

        with open(conversation_file_path2, "a") as file:
            file.write(f"user: {prompt}\n")
            file.write(f"bot: {response}\n\n")


# Knowledge Graph Page
if selected_page == "Knowledge Graph":
    st.title("Knowledge Graph")
    # Add your code to display the knowledge graph here
    
    st.write("This is the knowledge graph page.")

# Recommendations Page
if selected_page == "Recommendations":
    st.title("Recommended Food for your Disease!")

    # Display chat messages from Recommendations history on app rerun
    for message in st.session_state.messages_recommendations:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to Recommendations chat history
        st.session_state.messages_recommendations.append({"role": "user", "content": prompt})

        # Get model response
        response = graphRAG.get_response_recommedations(prompt)

        # Display model response in chat message container
        with st.chat_message("bot"):
            st.markdown(response)

        # Add model response to Recommendations chat history
        st.session_state.messages_recommendations.append({"role": "bot", "content": response})

        with open(conversation_file_path3, "a") as file:
            file.write(f"user: {prompt}\n")
            file.write(f"bot: {response}\n\n")