# ********** Defining API key **********
# Importing OpenAI key from a python file
from secret_keys import openai_key
import os
os.environ["OPENAI_API_KEY"] = openai_key

# ********** Creating LLM model object **********
# Defining model name
model_name = "gpt-3.5-turbo"
# Importing libraries from LangChain
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
# Making an object of the LLM model
llm = ChatOpenAI(model_name=model_name, temperature=0)

# ********** Asking questions to LLM Model **********
messages = [
    [
        SystemMessage(content="You are a helpful assistant that give good answers to specific questions.")
    ],
    [
        HumanMessage(content="Who is Shoaib Sikander?")
    ],
]
# Producing an answer
output=llm.generate(messages)
#print(output)
#print question and answer
# Extracting response from the answer
print('QUESTION: ' + messages[1][0].content)
print('ANSWER: ' + output.generations[1][0].message.content + '\n')
# Printing token usage and model name
print(output.llm_output)
print('\n\n')

# ********** Updating model's knowledge base with our own datat **********
# Importing libraries for loading a PDF file
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
# Loading PDF file
loader = UnstructuredPDFLoader('File.pdf')
documents = loader.load()
# Splitting the text loaded from document
SIZE=1000
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=SIZE, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(texts)
#print(len(texts))
# Loading the embeddings
#from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
# Saving into Vector Database
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings)
# Updating the model's knowledge base with new data
from langchain.chains import VectorDBQA
#from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
llm_updated = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=db, k=1)
#llm_updated = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=db, k=1)

# ********** Asking same question to the updated LLM model **********
question = "Who is Shoaib Sikander?"
output = llm_updated.invoke(question)
print('QUESTION: ' + output.get('query'))
print('ANSWER: ' + output.get('result'))