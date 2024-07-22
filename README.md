# Graph RAG for Medical Data

This project implements a Graph Retrieval-Augmented Generation (Graph RAG) system for medical data using a combination of LLM-based triplet extraction, knowledge graph construction, and semantic search capabilities.

## Description

The Graph RAG project leverages advanced LLM techniques to enhance medical data retrieval and understanding. Key functionalities include:

- **Triplet Generation**: Extracts and curates triplets from medical articles and books related to inflammation.
- **Knowledge Graph Construction**: Constructs a dynamic knowledge graph using Neo4j to represent extracted data.
- **Semantic Search**: Implements RAG techniques to query and retrieve relevant information from the knowledge graph.
- **Streamlit Application**: Provides an interactive interface to explore and query the constructed knowledge graph.

## Project Setup

### Environment Setup

1. **Clone the Repository**:

    ```bash
    git clone git@github.com:NivedhaBalakrishnan/End-to-end-GraphRAG.git
    cd End-to-end-GraphRAG
    ```

2. **Create a Virtual Environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:


    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

1. **Generate Triplet Data**:

    Run `generate_data.py` to extract triplets from your source texts:

    ```bash
    python generate_data.py
    ```

2. **Build Knowledge Graph**:

    Use `kgconstruct.py` to construct the knowledge graph in Neo4j:

    ```bash
    python kgconstruct.py
    ```

3. **Load Triplets into Neo4j**:

    Execute `GraphDB.py` to load the generated triplets into the Neo4j database:

    ```bash
    python GraphDB.py
    ```

4. **Run RAG Model**:

    Execute `GraphRAG.py` to perform retrieval-augmented generation using the Neo4j graph:

    ```bash
    python GraphRAG.py
    ```

5. **Start Streamlit Application**:

    Run `GraphQA.py` to launch the interactive Streamlit application for querying the knowledge graph:

    ```bash
    streamlit run GraphQA.py
    ```

6. **Add New Data**:

    If you need to add new data to the Neo4j database, use `Neo4j_append.py`:

    ```bash
    python Neo4j_append.py
    ```
