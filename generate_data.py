import os
import re
import pandas as pd
from helper import Helper
from model import GeneratorModel
from docutils import Document
from kgopenai import KGOpenai
import json

class GenerateData:
    def __init__(self):
        helper = Helper()
        helper.clear_log_file()
        self.log = helper.get_logger()

        self.model = GeneratorModel()
        # self.model = KGOpenai()
        self.doc = Document()
    

    def save_to_dataframe(self, data_list, df, base_name):
        try:
            n = len(df)

            rows = []
            for i, item in enumerate(data_list):
                item_dict = {'id': n+i, 'input': item['input'], 'output': item['output']}
                rows.append(item_dict)
            
            new_rows = pd.DataFrame(rows)
            self.log.info(f"New Rows: {new_rows}")
            df = pd.concat([df, new_rows], ignore_index=True)
            df.to_csv(f'output_data/{base_name}.csv', index=False)
            self.log.info("Successfully saved to dataframe!")
            return True
        except Exception as e:
            self.log.error(f"Error saving to dataframe: {e}")
            return False
        
    
    
    def post_process_data(self, data, save_path='output.json'):
        start_index = data.find("[")
        end_index = data.rfind("]") + 1

        if start_index != -1 and end_index != -1:
            # Extract JSON data from the string
            json_data_str = data[start_index:end_index]

            # Load existing JSON data from file if it exists
            if os.path.exists(save_path):
                with open(save_path, 'r') as json_file:
                    existing_json_data = json.load(json_file)
            else:
                existing_json_data = []

            # Load JSON data
            try:
                json_data = json.loads(json_data_str)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                json_data = []

            # Append new JSON data to existing data
            existing_json_data.extend(json_data)

            # Save updated JSON data back to the file
            with open(save_path, 'w') as json_file:
                json.dump(existing_json_data, json_file, indent=4)
        else:
            print("No JSON data found in the input.")
        # pattern = r"<output>(.*?)</output>"
    #     matches = re.findall(pattern, data, re.DOTALL)
    #     samples = []
    
    #     for match in matches:
    #         # match = match.strip().strip('\n')
    #         match = match.replace('\n', '')
    #         sample_dict = eval(match)
    #         samples.append(sample_dict)
    #     return samples 
        

    def generate_predictions(self, text, base_name, whatfor = 'generation'):
        try:
            if len(text.split()) > 4000:
                chunks = self.doc.chunk_documents(text)
                processed_output = []
                n = len(chunks)
                for i, chunk in enumerate(chunks):
                    df = self.get_dataframe(base_name)
                    self.log.info(f"Processing chunks {i+1}/{n}")
                    chunk_claude = self.model.model_prediction(chunk, whatfor)
                    processed_output = self.post_process_data(chunk_claude)
                    # self.save_to_dataframe(processed_output, df, base_name)
            else:
                df = self.get_dataframe()
                self.log.info(f"Processing text")
                claude = self.model.model_prediction(text, whatfor)
                processed_output = self.post_process_data(claude)
                # self.save_to_dataframe(processed_output, df, base_name)

            return True
        
        except Exception as e:
            self.log.error(f"Error generating predictions: {e}")
            return False

    def get_txt_files(self, directory):
        return [f for f in os.listdir(directory) if f.endswith('.txt')]


    def read_data(self, file):
        with open(file, 'r') as f:
            data = f.read()
        return data
    
    def get_dataframe(self, base_name):
        if os.path.exists(f'output_data/{base_name}.csv'):
            return pd.read_csv(f'output_data/{base_name}.csv')
        else:
            return pd.DataFrame(columns=['id', 'input', 'output'])

    def generate_training_data(self, directory):
        for i, file in enumerate(self.get_txt_files(directory)):
            base_name = file.split('.')[0]
            print(base_name)
            data = self.read_data(f"{directory}/{file}")
            self.log.info(f"Processing File: {i+1}")
            was_success = self.generate_predictions(data, base_name, whatfor = 'generation')
            if was_success:
                self.log.info(f"Successfully generated training data for file {i}!")
            else:
                self.log.error("Error generating training data!")

if __name__ == '__main__':
    generate_data = GenerateData()
    generate_data.generate_training_data('source')