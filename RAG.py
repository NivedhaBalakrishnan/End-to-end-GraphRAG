from vectorize import VectorDB
import anthropic
import os
import logging



# create log file
logging.basicConfig(filename='logs.log', filemode='w', level=logging.INFO)
open('logs.log', 'w').close()

class RAG():
    def __init__(self):
        try:
            API_KEY = os.getenv('ANTHROPIC_API_KEY')
            self.client = anthropic.Client(api_key=API_KEY)
        except Exception as e:
            self.log.error(f"Error connecting to the anthropic API {e}")
            raise e

    

    def get_completion(self, prompt, model='claude-2.1'):
        return self.client.completions.create(prompt=prompt, max_tokens_to_sample=4000, model=model).completion
        # max_tokens_to_sample = 4000 for optimal performance


    def model_prediction(self, question, text):
    # def model_prediction(self, question):
            prompt = "Your challenge is to respond to a question using the information provided in the accompanying text. Construct your answer by extracting relevant details from the text to ensure accuracy and completeness."
            input = f"""\n\nHuman: {prompt}
        
                            <question>{question}</question>

                            <text>{text}</text>
                            
                            Assistant:"""
            
            try:
                completion = self.get_completion(input)
                return completion
            except Exception as e:
                raise e

    def get_response(self,question):
        vector_db = VectorDB()
        docs = vector_db.get_db_retriever(question, k=3)
        logging.info(docs)
        text = [doc.page_content for doc in docs]
        logging.info(text)

        response = self.model_prediction(question, text)
        return response       


# if __name__ == "__main__":
#     model = LeximGPTModelClaude()
#     question = "Recommended food for inflammation."
#     response = model.get_response(question)
#     logging.info(response)
#     print(response)