import os
import logging
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Construct_KG:
    def __init__(self):
        load_dotenv()
        os.environ["NEO4J_URI"] = os.environ.get("NEO4J_URI")
        os.environ["NEO4J_USERNAME"] = os.environ.get("NEO4J_USERNAME")
        os.environ["NEO4J_PASSWORD"] = os.environ.get("NEO4J_PASSWORD")
        os.environ["NEO4J_DATABASE"] = os.environ.get("NEO4J_DATABASE") or "neo4j"
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

        print(os.environ["OPENAI_API_KEY"]) 


        self.graph = Neo4jGraph()
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["."])

        self.set_up_logging()
    

    def set_up_logging(self):
        # Clear the log file if it exists
        if os.path.exists('logs.log'):
            open('logs.log', 'w').close()
            
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a file handler
        file_handler = logging.FileHandler('logs.log')
        file_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the file handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)



    def convert_to_graph(self, documents):
        print("Converting to graph")
        graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
        self.logger.info(f"Converted documents to graph documents: {graph_documents}")
        self.graph.add_graph_documents(graph_documents)
    

    def get_documents(self, text):
        documents = self.text_splitter.split_text(text)
        self.logger.info(f"Split text into {len(documents)} documents")
        self.logger.info(f"Documents: {documents}")
        documents = [Document(text=doc, page_content=doc) for doc in documents]
        return documents





text = """In total, nine common lifestyle habits, including
dietary patterns, were identified among people
living in the Blue Zones. According to their
research, people that live the longest have
a diet composed mostly of plants.476 Leafy
greens like spinach and kale, seasonal fruits
and vegetables, and beans are the most
common foods eaten by people in the Blue
Zones.477 And other foods consumed by
these same people have also been identified
as effective in treating specific diseases.
For example, olive oil consumption, which
is prominent among middle-aged people in
Ikaria, has been shown to increase good
cholesterol and lower bad cholesterol.478
People with high levels of High-Density
Lipoprotein (HDL) or good cholesterol
have been shown to be at lower risk for
heart disease and stroke. Additionally,
egg consumption, which was found to be
low among all people living in the five Blue
Zones, has been linked to higher rates of
prostate cancer for men and aggravated
kidney problems for women. The research
demonstrates the strong impact food has on
overall health, and the link between physical
health with quality of life. """


kg = Construct_KG()
documents = kg.get_documents(text)
kg.convert_to_graph(documents)