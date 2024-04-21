# ********** Creating LLM model object **********
# A model is already downloaded and saved into working directory
llm_model_name = 'ggml-model-q4_0.gguf'
# Importing libraries from LangChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import LlamaCpp
# Making an object of the LLM model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(model_path=llm_model_name, temperature=0.0, top_p=1, n_ctx=6000, callback_manager=callback_manager, verbose=True)

# ********** Asking questions to LLM Model **********
# 1st method of asking questions to a LLM mode
question = "Who is Shoaib Sikander?"
answer = llm(question)
print('QUESTION: ', question)
print('ANSWER: ', answer)

# 2nd method of asking questions to a LLM model
question = """
Question: Who is Shoaib Sikander?
"""
answer = llm.invoke(question)
print('QUESTION: ', question)
print('ANSWER: ', answer)

# 3rd method of asking questions to a model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
prompt = PromptTemplate.from_template("What is {what}?")
chain = LLMChain(llm=llm, prompt=prompt)
answer = chain.run("Shoaib Sikander")
print('QUESTION: ', question)
print('ANSWER: ', answer)

# ********** Updating model's knowledge base with our own data **********
# Importing libraries for loading a PDF file
from langchain_community.document_loaders import PyPDFLoader
# Loading PDF file
loader = PyPDFLoader('File.pdf')
documents = loader.load()
#print(loader)
#print(documents)
# Splitting text loaded from document
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)
#print(all_splits)

# Loading the embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device': 'cpu'})

# Saving into Vector Database
from langchain_community.vectorstores import Chroma
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings)
vectordb2 = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="./abcde")
vectordb3 = Chroma(persist_directory="./abcde", embedding_function=embeddings)

# Performing Retrieval Augented Generation operation
from langchain.chains import RetrievalQA
llm_updated = RetrievalQA.from_chain_type(llm, retriever=vectordb3.as_retriever())

# ********** Asking question to updated LLM model **********
question = "Who is Shoaib Sikander?"
output = llm_updated({"query": question})
print('QUESTION: ' + output.get('query'))
print('ANSWER: ' + output.get('result'))