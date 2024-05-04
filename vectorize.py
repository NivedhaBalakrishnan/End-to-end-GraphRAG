import os
import sys
import logging
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader # for now
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import anthropic
from langchain_anthropic import ChatAnthropic


class VectorDB():
    def __init__(self):
        # Load and set the environment variable
        load_dotenv()
        API_KEY = os.getenv('ANTHROPIC_API_KEY') # Authentication
        self.llm = ChatAnthropic(model="claude-2.1", anthropic_api_key=API_KEY)
        # Retrieve and initialize LLM model
        # llama2_13b = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
        # self.llm = Replicate(model=llama2_13b, model_kwargs={"temperature": 0.5, "top_p": 0.8})

        # encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
        self.embedding_function = HuggingFaceEmbeddings()    #encode_kwargs=encode_kwargs

        logging.basicConfig(filename='logs.log', filemode='w', level=logging.INFO)
        logging.info('Loaded the model!')
        sys.stdout.flush()


    def chunk_documents(self):
        loader = DirectoryLoader('./source/', glob="./*.txt", loader_cls=TextLoader)
        documents = loader.load()

        # Chunk the data
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""], chunk_size=1000, chunk_overlap=200)
        chunked_documents = text_splitter.split_documents(documents)
        logging.info('chunking')
        logging.info(chunked_documents)

        for i in chunked_documents:
            logging.info(i.metadata)
        return(chunked_documents)
    

    def push_to_database(self, chunked_documents):
        
        # Create vector database
        vectorstore = Chroma.from_documents(documents=chunked_documents, embedding=self.embedding_function, persist_directory="./chromadb")
        vectorstore.persist()
        vectorstore = None
        print('pushed')
    
    
    def get_db_retriever(self, question,k=3):
        vectorstore = Chroma(persist_directory="./chromadb", 
                  embedding_function=self.embedding_function)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        return docs
    
    

# if __name__ == "__main__":
#     rag = VectorDB()
#     chunked_documents = rag.chunk_documents()
#     rag.push_to_database(chunked_documents)

    