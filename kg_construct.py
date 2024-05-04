from neo4j import GraphDatabase
import json
import os
from dotenv import load_dotenv


load_dotenv()

# Connect to Neo4j database
uri = os.environ.get("NEO4J_URI")
user = os.environ.get("NEO4J_USERNAME")
password = os.environ.get("NEO4J_PASSWORD")
print(uri, user, password)
driver = GraphDatabase.driver(uri, auth=(user, password))

def create_nodes_and_relationships(data):
    with driver.session() as session:
        for item in data:
            # Extract the keys and values
            keys = list(item.keys())
            values = list(item.values())
            print(keys, values)

            if None in values or len(values) < 3 or len(keys) < 3:
                continue

            # Create nodes for keys other than "Relationship"
            create_node1 = f"""
            MERGE (n1:`{keys[0]}` {{id: $value1}})
            RETURN n1
            """
            node1 = session.execute_write(lambda tx: tx.run(create_node1, value1=values[0]).single())

            create_node2 = f"""
            MERGE (n2:`{keys[2]}` {{id: $value2}})
            RETURN n2
            """
            node2 = session.execute_write(lambda tx: tx.run(create_node2, value2=values[2]).single())

            # Create the relationship between the nodes
            relation = values[1].replace(" ", "_")
            create_relationship = f"""
            MATCH (n1:`{keys[0]}` {{id: $value1}}), (n2:`{keys[2]}` {{id: $value2}})
            MERGE (n1)-[:{relation}]->(n2)
            RETURN n1, n2
            """
            session.execute_write(lambda tx: tx.run(create_relationship, value1=values[0], value2=values[2]))


# Example usage
with open("output.json", "r") as read_file:
    json_data = json.load(read_file)

create_nodes_and_relationships(json_data)
driver.close()