from neo4j import GraphDatabase
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
import textwrap
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain




class NLQ():
    def __init__(self) -> None:
        load_dotenv()
        NEO4J_URI = os.environ.get("NEO4J_URI")
        NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
        NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
        self.graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        API_KEY = os.getenv('ANTHROPIC_API_KEY') # Authentication
        self.chain = GraphCypherQAChain.from_llm(
            ChatAnthropic(model="claude-2.1", anthropic_api_key=API_KEY), graph=self.graph, verbose=True)
    
    def get_response(self, question):
        response = self.chain.run("Recommended food for hypertension?")
        return response


class GraphEmbeddings():
    def __init__(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        load_dotenv()
        self.NEO4J_URI = os.environ.get("NEO4J_URI")
        self.NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
        self.NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.NEO4J_URI, auth=(self.NEO4J_USERNAME, self.NEO4J_PASSWORD))
        self.embedding_function = HuggingFaceEmbeddings()
        self.index_name = "nodeEmbeddings"
        self.embedding_dimensions = 768
        


    def retrieve_nodes_and_relationships(self, tx, node_id):
        query = """
        MATCH (n)-[r]->(m)
        WHERE n.id = $node_id
        RETURN n.id AS node1, type(r) AS relationship, m.id AS node2, '->' AS direction
        UNION
        MATCH (n)<-[r]-(m)
        WHERE n.id = $node_id
        RETURN n.id AS node1, type(r) AS relationship, m.id AS node2, '<-' AS direction
        """
        result = tx.run(query, node_id=node_id)
        return [(record["node1"], record["relationship"], record["node2"], record["direction"]) for record in result]

    def generate_embeddings(self):
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN n as node").data()
            for node in nodes:
                node_id = node['node']["id"]
                data = session.read_transaction(self.retrieve_nodes_and_relationships, node_id)
                paragraph = ""
                for node1, relationship, node2, direction in data:
                    sentence = f"{node1} {direction} {relationship} {direction} {node2}."
                    paragraph += sentence + " "
                embedding = self.embedding_function.embed_query(paragraph.strip())
                session.run(
                    "MATCH (n {id: $node_id}) SET n.embedding = $embedding, n.info = $paragraph",
                    node_id=node_id,
                    embedding=list(embedding),
                    paragraph=paragraph.strip()
                )
        
    
    def create_vector_index(self):
        with self.driver.session() as session:
            session.run("""
            CREATE VECTOR INDEX nodeEmbeddings IF NOT EXISTS
            FOR (node:id) ON (node.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 768,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """)

    def close(self):
        self.driver.close()

    
    def get_retriever(self):
        try: 
            neo4j_vector_store = Neo4jVector.from_existing_graph(
                embedding = self.embedding_function,
                url=self.NEO4J_URI,
                username=self.NEO4J_USERNAME,
                password=self.NEO4J_PASSWORD,
                database="neo4j",
                index_name="nodeEmbeddings",
                node_label="id",
                embedding_node_property="embedding",
                text_node_properties=["id","info"]
            )
            retriever = neo4j_vector_store.as_retriever()
            print('Retriever created successfully')
            return retriever
        except Exception as e:
            print(f'Error creating retriever: {e}')
            raise e
    
    def get_chain(self):
        try:
            retriever = self.get_retriever()
            
            if retriever:
                API_KEY = os.environ.get('ANTHROPIC_API_KEY')
                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    ChatAnthropic(model="claude-2.1", anthropic_api_key=API_KEY),
                    chain_type="stuff",
                    retriever=retriever, return_source_documents=True)

                print('Chain created successfully')
                return chain
            else:
                print('Retriever not created')
                return None
        except Exception as e:
            print(f'Error creating chain: {e}')
            raise e
    
    def get_response(self, question):
        chain = self.get_chain()
        try:
            """Pretty print the chain's response to a question"""
            response = chain.invoke({"question": question}, verbose=True)
            print(response)
            # sources = response.get('source_documents', '')
            # source_content = [doc.page_content for doc in sources]
            # return textwrap.fill(response['answer'], 60)
        except Exception as e:
            print(f'Error pretty printing chain: {e}')
            raise e

        
# Usage example
graph_embeddings = GraphEmbeddings()
# graph_embeddings.generate_embeddings()
# graph_embeddings.create_vector_index()
# graph_embeddings.close()
question = "Diet for rheumatoid pain."
response = graph_embeddings.get_response(question)
# print(response)