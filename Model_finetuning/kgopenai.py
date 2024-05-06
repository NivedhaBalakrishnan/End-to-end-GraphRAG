import re
from helper import Helper
from docutils import Document
from openai import OpenAI




class KGOpenai():
    def __init__(self):
        hlp = Helper()
        #hlp.clear_log_file()  #to clear history
        self.log = hlp.get_logger()

        try:
            self.openai_client = OpenAI()
            self.log.info("Successfully connected to the OPENAI API")
        except Exception as e:
            self.log.error(f"Error connecting to the OPENAI API {e}")
            raise e
        


    def get_prompt(self, whatfor='process'):
        try:
            with open(f"prompt/{whatfor}.txt", 'r') as file:
                prompt = file.read()
            return prompt
        except FileNotFoundError:
            raise FileNotFoundError
        except Exception as e:
            self.log.error(f"An error occurred: {e}")
            raise e

    
    def model_prediction(self, text, whatfor='process'):
        prompt = self.get_prompt(whatfor)
        
        input = f"""{prompt}
                    <article>{text}</article>
                    """
        
        try:
            output = self.openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            temperature=0.2,
                            messages=
                            [
                                {"role": "user",
                                "content": input
                                }
                            ]
                            ).choices[0].message.content
            self.log.info(f"PREDICTION: {output}")
            return output
        except Exception as e:
            self.log.error(f"Error generating Prediction: {e}")
            raise e