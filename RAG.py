from vectorize import VectorDB
import anthropic
import os
import logging
import pandas as pd
from eval import Evaluation
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


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
    
    def get_response_eval(self,question):
        vector_db = VectorDB()
        docs = vector_db.get_db_retriever(question, k=3)
        logging.info(docs)
        text = [doc.page_content for doc in docs]
        logging.info(text)

        response = self.model_prediction(question, text)
        print(response)
        return question, text, response
    
    def get_evaluated(self, question, source, answer):

            evaluation = Evaluation()
            # embeddings_model = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))
            embeddings_model = HuggingFaceEmbeddings()
            # similarity_score = self.evaluation.evaluate_similarity(
            #     embeddings_model,
            #     answer,
            #     source
            # )
            groundedness_score = evaluation.groundedness(question, source, answer)
            context_relevancy_score = evaluation.context_relevancy(question, source, answer)
            score, reason = evaluation.evaluate(question, source, answer)
            # coherence_score, coherence_reason = evaluation.evaluate_coherence(question, answer)
            faithfulness_score, faithfulness_reason = evaluation.evaluate_faithfulness(question, answer, source)
            # #contextual_precision_score, contextual_precision_reason = evaluation.evaluate_contextual_precision(question, answer, source)
            # #contextual_recall_score, contextual_recall_reason = evaluation.evaluate_contextual_recall(question, answer, source)
            hallucination_score, hallucination_reason = evaluation.evaluate_hallucination(question, answer, source)
            # toxicity_score, toxicity_reason = evaluation.evaluate_toxicity(question, answer)
            # bias_score, bias_reason = evaluation.evaluate_bias(question, answer)
            # #ragas_score = evaluation.evaluate_ragas(question, answer, source)

            # return (similarity_score, score, reason, coherence_score, coherence_reason, faithfulness_score,
            #         faithfulness_reason, hallucination_score, hallucination_reason,
            #         toxicity_score, toxicity_reason, bias_score, bias_reason)
    

    def save_to_csv(self, question, source, answer, scores):

        data = {
            'Question': question,
            'Source': source,
            'Answer': answer,
            'Similarity Score': scores[0],
            'Relevancy Score': scores[1],
            'Reason': scores[2],
            'Coherence Score': scores[3],
            'Coherence Reason': scores[4],
            'Faithfulness Score': scores[5],
            'Faithfulness Reason': scores[6],
            'Hallucination Score': scores[7],
            'Hallucination Reason': scores[8],
            'Toxicity Score': scores[9],
            'Toxicity Reason': scores[10],
            'Bias Score': scores[11],
            'Bias Reason': scores[12]
        }

        df = pd.DataFrame([data])

        file_exists = os.path.isfile('er_metrics.csv')

        if file_exists:
            df.to_csv('er_metrics.csv', mode='a', header=False, index=False)
        else:
            df.to_csv('er_metrics.csv', index=False)

        print("Scores saved successfully.")




if __name__ == '__main__':
    rag = RAG()
    questions = [
    "What are some of the key foods and strategies focused on in an anti-inflammatory diet?",
    "Why is reducing consumption of processed and red meat recommended for an anti-inflammatory eating approach?",
    "What role does achieving a healthy weight play in reducing inflammation in the body?",
    "How can probiotics and prebiotics help improve gut health and reduce inflammation?",
    "What are the recommended omega-6 to omega-3 ratios to aim for in an anti-inflammatory diet?",
    "Why are whole foods preferred over isolating specific nutrients in an anti-inflammatory eating plan?",
    "What are some of the health conditions that research indicates may benefit from following an anti-inflammatory diet?",
    "How can herbs and spices like turmeric, ginger and rosemary help reduce inflammation when included regularly in the diet?",
    "What are the potential benefits of plant-based proteins like legumes, nuts and seeds compared to animal proteins in terms of inflammation?",
    "Why is it important to focus on foods to include more of rather than just foods to restrict or avoid in an anti-inflammatory eating approach?"
    ]
    total_elapsed_time = 0
    question_count = 0
    for question in questions:
        answer, source, elapsed_time = rag.get_response_eval(question)
        print(f"A: {answer}")
        scores = rag.get_evaluated(question, source, answer)
        rag.save_to_csv(question, source, answer, scores)

        total_elapsed_time += elapsed_time
        question_count += 1

    average_time = total_elapsed_time / question_count
    print(f'Average time per question: {average_time} seconds')



# if __name__ == "__main__":
#     model = LeximGPTModelClaude()
#     question = "Recommended food for inflammation."
#     response = model.get_response(question)
#     logging.info(response)
#     print(response)