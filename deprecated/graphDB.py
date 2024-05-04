import os
from helper import Helper

import re
import json
from datetime import datetime
from dotenv import load_dotenv 


from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings

from kgopenai import KGOpenai
from docutils import Document

from bs4 import BeautifulSoup






class GraphDBNeo4J():
    def __init__(self):
        load_dotenv('.env', override=True)
        hlp = Helper()
        hlp.clear_log_file()  #to clear history
        self.log = hlp.get_logger()

        NEO4J_URI = os.getenv('NEO4J_URI')
        NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
        
        self.kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'

        self.VECTOR_INDEX_NAME = 'article_chunk'
        self.VECTOR_NODE_LABEL = 'Chunk'
        self.VECTOR_SOURCE_PROPERTY = 'text'
        self.VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'

        self.model = KGOpenai()

        self.doc = Document()

    
    

    def show_index(self):
        index = self.kg.query("SHOW INDEXES")
        self.log.info(f"Index: {index}")
        return index



    # def create_graph_nodes(self, chunk):
    #     merge_chunk_node_query = """
    #                             MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
    #                             ON CREATE SET
    #                                 mergedChunk.source = $chunkParam.source,
    #                                 mergedChunk.authors = $chunkParam.authors,
    #                                 mergedChunk.journal = $chunkParam.journal,
    #                                 mergedChunk.publicationdate = $chunkParam.publicationdate,
    #                                 mergedChunk.summary = $chunkParam.summary,
    #                                 mergedChunk.text = $chunkParam.text,
    #                                 mergedChunk.chunkSeqId = $chunkParam.chunkSeqId,
    #                                 mergedChunk.article = $chunkParam.article,
    #                                 mergedChunk.about = $chunkParam.about
    #                             RETURN mergedChunk
    #                             """
        
    #     try:
    #         self.kg.query(merge_chunk_node_query, params={"chunkParam": chunk})
    #         self.kg.query("""CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
    #                       FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE""")
    #         # self.log.info(f"Created graph nodes")

    #     except Exception as e:
    #         self.log.error(f"An error occurred: {e}")
    #         raise e

    
    
    
    def create_constraints(self):
        try:
            self.kg.query("""CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
                            FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE""")
            # self.log.info(f"Created constraint")
        except Exception as e:
            self.log.error(f"Index not created: {e}")
            raise e
    


    def get_number_of_nodes(self):
        node_count_query = """
                            MATCH (n:Chunk)
                            RETURN count(n) as nodeCount
                            """
        node_count = self.kg.query(node_count_query)[0]['nodeCount']
        self.log.info(f"Number of nodes: {node_count}")
        return node_count


    def create_vector_index(self):
        try:
            self.kg.query("""
            CREATE VECTOR INDEX `article_chunk` IF NOT EXISTS
            FOR (c:Chunk) ON (c.textEmbedding) 
            OPTIONS { indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'    
            }}
            """)
            # self.log.info(f"Created vector index")
        except Exception as e:
            self.log.error(f"Vector index not created: {e}")
            raise e

    
    def calculate_embeddings(self):
        try:
            # Create an instance of OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)

            # Retrieve all Chunk nodes without embeddings
            chunks = self.kg.query("""
                MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
                RETURN chunk
            """)
            for chunk in chunks:
                # Calculate the embedding for the chunk text
                embedding = embeddings.embed_query(chunk['chunk']['text'])

                # Update the Chunk node with the calculated embedding
                self.kg.query("""
                    MATCH (chunk:Chunk {chunkId: $chunkId})
                    SET chunk.textEmbedding = $embedding
                """, params={"chunkId": chunk['chunk']['chunkId'], "embedding": embedding})

            # self.log.info(f"Calculated embeddings")
        except Exception as e:
            self.log.error(f"Embeddings not calculated: {e}")
            raise e


    def create_article_info(self, article):
        try:
            cypher = """
                    MATCH (anyChunk:Chunk) 
                    WHERE anyChunk.article = $article
                    WITH anyChunk LIMIT 1
                    RETURN anyChunk { .source, .authors, .journal, .publicationdate, .summary, .article, .about} as article_info
                    """
            article_info = self.kg.query(cypher, params={'article': article})
            self.log.info(f"Article info created: {article_info}")
            return article_info[0]["article_info"]
        except Exception as e:
            self.log.error(f"Article info not created: {e}")
            raise e
    

    def merge_article_node(self, article_info):
        try:
            cypher = """
                MERGE (f:Article {article: $ArticleInfoParam.article })
                ON CREATE 
                    SET f.authors = $ArticleInfoParam.authors
                    SET f.source = $ArticleInfoParam.source
                    SET f.journal = $ArticleInfoParam.journal
                    SET f.publicationdate = $ArticleInfoParam.publicationdate
                    SET f.summary = $ArticleInfoParam.summary
                    SET f.about = $ArticleInfoParam.about
            """

            self.kg.query(cypher, params={'ArticleInfoParam': article_info})
            self.log.info(f"Article node merged")
        except Exception as e:
            self.log.error(f"Article node not merged: {e}")
            raise e


    def create_relationship(self, article_info):
        try:
            cypher = """
                    MATCH (from_same_article:Chunk)
                    WHERE from_same_article.article = $articleIdParam
                    WITH from_same_article
                        ORDER BY from_same_article.chunkSeqId ASC
                    WITH collect(from_same_article) as section_chunk_list
                        CALL apoc.nodes.link(
                            section_chunk_list, 
                            "NEXT", 
                            {avoidDuplicates: true}
                        )
                    RETURN size(section_chunk_list)
                    """
            self.kg.query(cypher, params={'articleIdParam': article_info['article']})
            self.log.info(f"Relationship created")
        except Exception as e:
            self.log.error(f"Relationship not created: {e}")
            raise e
    

    def connext_to_parent(self):
        try:
            cypher = """
                    MATCH (c:Chunk), (f:Article)
                        WHERE c.article = f.article
                    MERGE (c)-[newRelationship:PART_OF]->(f)
                    RETURN count(newRelationship)
                """
            self.kg.query(cypher)
            self.log.info(f"Connected to parent")
        except Exception as e:
            self.log.error(f"Connection to parent not created: {e}")
            raise e



    def connect_all_article_bidirectional(self):
        try:
            # Cypher query to match and connect all forms bidirectionally
            cypher_query = """
                MATCH (a:Article), (b:Article)
                WHERE a <> b AND a.about = b.about
                MERGE (a)-[:CONNECTED_TO]->(b)
                MERGE (b)-[:CONNECTED_TO]->(a)
                RETURN count(*) AS connections
            """
            result = self.kg.query(cypher_query)
            connections_count = result[0]['connections']
            self.log.info(f"Established bidirectional connections between all forms: {connections_count}")
        except Exception as e:
            self.log.error(f"Error while establishing bidirectional connections: {e}")
            raise e



    def process(self, directory):
        files = os.listdir(directory)
        
        for file in files:
            file_path = os.path.join(directory, file)
            json_data = self.get_json_data(file_path)
            file_name = os.path.basename(file_path.split('.')[-2])
            self.log.info(f"Processing file: {file_name}")
            chunked_with_metadata = self.create_chunks(json_data, file_name)
            # self.log.info(f"Chunked data: {chunked_with_metadata}")

            self.create_vector_index()
            for chunk in chunked_with_metadata:
                self.create_graph_nodes(chunk)
                self.calculate_embeddings()
            article_info = self.create_article_info(chunk["article"])
            self.merge_article_node(article_info)
            self.create_relationship(article_info)
            self.connext_to_parent()

        self.connect_all_article_bidirectional()
        self.kg.refresh_schema()
        print(self.kg.schema)
        total_node_count = self.get_number_of_nodes()
        self.log.info(f"Total number of nodes: {total_node_count}")
    
#------------------------------------------------------------------------------------------------------------#
    def create_chunks(self, json_data, filename, chunk_seq_id=0):
        chunks_with_metadata = []
        chunks = self.doc.chunk_documents(json_data['text'])
        self.log.info(f"Text split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            json_data['text'] = chunk
            json_data['chunkSeqId'] = i
            chunk_id = f"{filename}_{chunk_seq_id}"
            json_data['chunkId'] = chunk_id
            chunks_with_metadata.append(json_data.copy())
            chunk_seq_id += 1
        return chunks_with_metadata
    



    def get_json_data(self, file_path):
        try:
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                return json_data
        except FileNotFoundError:
            self.log.error("File not found")
            raise FileNotFoundError
        except Exception as e:
            self.log.error(f"An error occurred: {e}")
            raise e



    def extract_dict_content(self, text):
        pattern = r'\{[^{}]*\}'
        match = re.search(pattern, text)

        if match:
            dict_content = match.group()
            return dict_content
        


    def get_metadata(self, text):
        metadata = None
        attempt_count = 0
        max_attempts = 3

        while not metadata and attempt_count < max_attempts:
            metadata = self.model.model_prediction(text, whatfor='metadata')

            attempt_count += 1
            try:
                dict_content = self.extract_dict_content(metadata)
                if dict_content:
                    metadata = json.loads(dict_content)
                    if not isinstance(metadata, dict):
                        metadata = None
                        print(f"Attempt {attempt_count}: Extracted content is not a valid dictionary. Retrying...")
                else:
                    print(f"Attempt {attempt_count}: No dictionary content found in the LLM output. Retrying...")
            except json.JSONDecodeError:
                print(f"Attempt {attempt_count}: Extracted content is not a valid JSON string. Retrying...")

        if metadata:
            dict_data = metadata

            return dict_data

        return {}
    


    def save_to_json(self, data, output_dir, base_name):
        with open(f'{output_dir}/{base_name}.json', 'w') as file:
            json.dump(data, file, indent=4)
        self.log.info(f"Data saved to {output_dir}/{base_name}.json")
    


    def post_process_data(self, data):
        self.log.info(f"Data: {data}")
        soup = BeautifulSoup(data, 'html.parser')
        # Find the <output> tag
        output_tag = soup.find('output')
        if output_tag:
            # Extract the text inside the <output> tag
            output_text = output_tag.get_text(strip=True)
            # Check if the </output> tag is present
            if output_text.endswith('</output>'):
                # Evaluate the text as a Python list
                try:
                    output_list = eval(output_text.replace('</output>', ''))
                    return output_list
                except (SyntaxError, NameError, TypeError, ZeroDivisionError):
                    print("Error: Invalid list format inside <output> tag.")
            else:
                # If </output> is missing, collect dictionaries from the text
                pattern = r"\{'entity1': '([^']+)', 'relationship': '([^']+)', 'entity2': '([^']+)'\}"
                output_list = [{'entity1': m.group(1), 'relationship': m.group(2), 'entity2': m.group(3)}
                            for m in re.finditer(pattern, output_text)]
                return output_list
        else:
            print("No <output>  tag found in the input text.")
        return []



    def extract_entities(self, chunk_text):
        predicted_entities = self.model.model_prediction(chunk_text, whatfor='process')
        output_list = self.post_process_data(predicted_entities)
        return output_list
    

    def merge_entities_to_chunk(self, chunk, entities_list):
        merged_chunks = []
        for i, entity_json in enumerate(entities_list):
            for j, entity in enumerate([entity_json['entity1'], entity_json['entity2']]):
                entity_json_merged = chunk.copy()
                entity_json_merged['entity'] = entity
                entity_json_merged['entityId'] = f"{chunk['chunkId']}_{i}_{j}"
                merged_chunks.append(entity_json_merged)
        return merged_chunks


    def create_entity_nodes(self, merged_chunk):
        create_entity_node_query = """
            MERGE (e:Entity {name: $name})
            ON CREATE SET
                e.entityId = $entityId,
                e.chunkId = $chunkId
            ON MATCH SET
                e.chunkId = $chunkId
        """
        params = {
            "entityId": merged_chunk['entityId'],
            "name": merged_chunk['entity'],
            "chunkId": merged_chunk['chunkId']
        }
        result = self.kg.query(create_entity_node_query, params=params)
        self.log.info(f"Created entity node: {merged_chunk['entity']}")
        

    def calculate_entity_embeddings(self, merged_chunk):
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)
        embedding = embeddings.embed_query(merged_chunk['entity'])
        update_entity_embedding_query = """
            MATCH (e:Entity {entityId: $entityId})
            SET e.embedding = $embedding
        """
        params = {
            "entityId": merged_chunk['entityId'],
            "embedding": embedding
        }
        self.kg.query(update_entity_embedding_query, params=params)
        self.log.info(f"Calculated embedding for entity: {merged_chunk['entity']}")
    

    def connect_entities(self, entities_list):
        for entity_json in entities_list:
            create_relationship_query = """
                MATCH (e1:Entity), (e2:Entity)
                WHERE toLower(e1.name) = toLower($entity1) AND toLower(e2.name) = toLower($entity2)
                MERGE (e1)-[r:{relationship_label}]->(e2)  // Replace :RELATION with dynamic label
                ON CREATE SET r.name = $relationship
            """
            params = {
                "entity1": entity_json['entity1'].lower(),
                "entity2": entity_json['entity2'].lower(),
                "relationship": entity_json['relationship'],
                "relationship_label": entity_json['relationship'].replace(' ', '_')  # Sanitize label if necessary
            }
            create_relationship_query = create_relationship_query.format(relationship_label=params["relationship_label"])
            self.kg.query(create_relationship_query, params=params)
            self.log.info(f"Connected entities: {entity_json['entity1']} - {entity_json['relationship']} - {entity_json['entity2']}")


    def data_to_graphdb(self, directory):
        json_dir = "json_data"
        files = os.listdir(directory)
        for i, file in enumerate(files):
            print(f"Processing file: {file}")
            base_name = file.split('.')[0]
            if os.path.exists(f'{json_dir}/{base_name}.json'):
                with open(f'{json_dir}/{base_name}.json', 'r') as file:
                    data_json = json.load(file)
                    print(f"Data loaded from {json_dir}/{base_name}.json")
            else:
                file_path = os.path.join(directory, file)
                with open(file_path, 'r') as file:
                    text = file.read()
                    data_json = self.get_metadata(text)

                    # Add unique document ID and text to the json
                    timestamp = datetime.now()
                    data_json['documentId'] = 'Document_' + str(i) + timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    data_json['text'] = text

                    # Save metadata from the text
                    self.save_to_json(data_json, json_dir, base_name)

            if data_json:
                
                chunked_data = self.create_chunks(data_json, base_name)
                for i, chunk in enumerate(chunked_data):
                        entities_list = self.extract_entities(chunk['text'])
                        self.log.info(f"Entities extracted: {entities_list}")
                        # entities_list = [{'entity1': 'Inflammation', 'relationship': 'effects on', 'entity2': 'B-Cell Development'}, {'entity1': 'B lymphocytes', 'relationship': 'components of', 'entity2': 'innate immunity'}, {'entity1': 'B lymphocytes', 'relationship': 'components of', 'entity2': 'adaptive immunity'}, {'entity1': 'B lymphocytes', 'relationship': 'role in', 'entity2': 'specific removal of pathogens'}, {'entity1': 'B lymphocytes', 'relationship': 'role in', 'entity2': 'specific removal of toxins'}, {'entity1': 'B lymphocytes', 'relationship': 'focus of', 'entity2': 'study'}, {'entity1': 'B-cell antigen receptors', 'relationship': 'generation of', 'entity2': 'specificity'}, {'entity1': 'B-cell antigen receptors', 'relationship': 'generation of', 'entity2': 'affinity'}, {'entity1': 'B-cell development', 'relationship': 'division into', 'entity2': 'phases'}, {'entity1': 'B-cell development', 'relationship': 'focus of', 'entity2': 'study'}, {'entity1': 'B-cell development', 'relationship': 'impact of', 'entity2': 'inflammation'}, {'entity1': 'B-cell development', 'relationship': 'impact of', 'entity2': 'acute inflammation'}, {'entity1': 'B-cell development', 'relationship': 'impact of', 'entity2': 'chronic inflammation'}, {'entity1': 'B-cell development', 'relationship': 'overview of', 'entity2': 'processes'}, {'entity1': 'B-cell development', 'relationship': 'overview of', 'entity2': 'bone marrow'}, {'entity1': 'B-cell development', 'relationship': 'overview of', 'entity2': 'periphery'}, {'entity1': 'B-cell development', 'relationship': 'impact of', 'entity2': 'acute inflammation'}, {'entity1': 'B-cell development', 'relationship': 'impact of', 'entity2': 'chronic inflammation'}, {'entity1': 'B-cell development', 'relationship': 'impact of', 'entity2': 'processes'}, {'entity1': 'B-cell development', 'relationship': 'impact of', 'entity2': 'bone marrow'}, {'entity1': 'B-cell development', 'relationship': 'impact of', 'entity2': 'periphery'}, {'entity1': 'B-cell development', 'relationship': 'division into', 'entity2': 'phases'}, {'entity1': 'B-cell development', 'relationship': 'impact of', 'entity2': 'acute inflammation'}, {'entity1': 'B-cell development', 'relationship': 'impact of', 'entity2': 'chronic inflammation'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'molecular events'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'bone marrow'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'developmental checkpoints'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'B lineage'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'lymphoid/myeloid branch point'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'multipotent progenitors'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'adhesion molecule'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'B-cell specific transcription factors'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'IL-7 receptor signaling'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'transcriptional networks'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'pro-B cells'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'transcriptional networks'}, {'entity1': 'B-cell development', 'relationship': 'focus on', 'entity2': 'PU.1 and E2A'}]
                        merged_chunks = self.merge_entities_to_chunk(chunk, entities_list)
                        for merged_chunk in merged_chunks:
                            self.create_entity_nodes(merged_chunk)
                            self.calculate_entity_embeddings(merged_chunk)

                        self.connect_entities(entities_list)
                        
        



if __name__ == '__main__':
    kg = GraphDBNeo4J()
    #Input: text file
    #Output: JSON data - Node: Entity => UniqueID, Properties: source, authors, journal, publicationDate, articleId, textEmbedding, chunkSeqId
    directory = "source_data"
    kg.data_to_graphdb(directory)