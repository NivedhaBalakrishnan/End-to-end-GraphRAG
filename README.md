# Graph RAG for Medical Data Using LLM

## Overview

This project demonstrates a comprehensive approach to building a Graph Retrieval-Augmented Generation (RAG) system for medical data. It leverages Large Language Models (LLMs) to generate knowledge triplets from medical literature and integrates them into a Neo4j graph database for advanced querying and information retrieval.

## Project Structure

- **KGOpenai.py**: Generates triplets from medical articles using an LLM. This script extracts meaningful relationships and entities from text to form structured knowledge.
  
- **GraphDB.py**: Utilizes `KGOpenai.py` to convert the extracted triplets into JSON format, which is then loaded into a Neo4j database. This script sets up the graph database with the initial data.
  
- **GraphRAG.py**: Implements the Retrieval-Augmented Generation (RAG) approach using the Neo4j database. It enables querying the graph for relevant information to enhance the response generation.
  
- **Neo4j_append.py**: Provides functionality to add new data to the Neo4j database. This script is useful for expanding the knowledge base with additional information.
  
- **GraphQA.py**: A Streamlit application that interfaces with the RAG system. It allows users to query the graph database and get responses generated based on the retrieved data.

## Usage

1. **Generate Triplets**: Run `KGOpenai.py` to extract triplets from medical texts.
   
2. **Populate Neo4j**: Use `GraphDB.py` to load the triplets into the Neo4j database.
   
3. **Query and Generate Responses**: Execute `GraphRAG.py` to perform retrieval and generation tasks using the graph database.
   
4. **Expand the Database**: Use `Neo4j_append.py` to add more data as needed.
   
5. **Interact with the System**: Launch `GraphQA.py` to use the Streamlit app for querying and interacting with the RAG system.

## Requirements

- Python 3.x
- Neo4j
- Streamlit
- Required Python libraries (listed in `requirements.txt`)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
