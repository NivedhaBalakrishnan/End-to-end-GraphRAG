import os
from helper import Helper
# Common data processing
import json
import textwrap
import pandas as pd

from dotenv import load_dotenv 

# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from neo4j import GraphDatabase
import openai

import time



class GraphRAG():
    def __init__(self) -> None:

        load_dotenv('.env', override=True)
        hlp = Helper()
        hlp.clear_log_file()
        self.log = hlp.get_logger()
        
        self.NEO4J_URI = os.getenv('NEO4J_URI')
        self.NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        self.NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'

        self.VECTOR_INDEX_NAME = 'article_chunk'
        self.VECTOR_NODE_LABEL = 'Chunk'
        self.VECTOR_SOURCE_PROPERTY = 'text'
        self.VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'

  
    
    def get_retriever(self):
        try: 
            neo4j_vector_store = Neo4jVector.from_existing_graph(
                embedding = OpenAIEmbeddings(api_key=self.OPENAI_API_KEY),
                url=self.NEO4J_URI,
                username=self.NEO4J_USERNAME,
                password=self.NEO4J_PASSWORD,
                database=self.NEO4J_DATABASE,
                index_name=self.VECTOR_INDEX_NAME,
                node_label=self.VECTOR_NODE_LABEL,
                embedding_node_property=self.VECTOR_EMBEDDING_PROPERTY,
                text_node_properties=[self.VECTOR_SOURCE_PROPERTY]
            )
            retriever = neo4j_vector_store.as_retriever()
            self.log.info('Retriever created successfully')
            return retriever
        except Exception as e:
            self.log.error(f'Error creating retriever: {e}')
            raise e



    def get_chain(self):
        try:
            retriever = self.get_retriever()
            
            if retriever:
                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    ChatOpenAI(temperature=0.5),
                    chain_type="stuff",
                    retriever=retriever, return_source_documents=True)

                self.log.info('Chain created successfully')
                return chain
            else:
                self.log.error('Retriever not created')
                return None
        except Exception as e:
            self.log.error(f'Error creating chain: {e}')
            raise e
    

    def get_response(self, question):
        chain = self.get_chain()
        try:
            """Pretty print the chain's response to a question"""
            start_time = time.time()
            response = chain.invoke({"question": question})

            end_time = time.time()  # Store the end time
            elapsed_time = end_time - start_time

            sources = response.get('source_documents', '')
            source_content = [doc.page_content for doc in sources]
            print(f"Source: {source_content}")
            return textwrap.fill(response['answer'], 60), source_content, elapsed_time
        except Exception as e:
            self.log.error(f'Error pretty printing chain: {e}')
            raise e


    