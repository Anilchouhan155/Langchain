import os
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WebBaseLoader

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)


os.environ['serpapi_api_key']="YOUR_serpapi_api_key"
os.environ['OPENAI_API_KEY']="YOUR_OPENAI_API_KEY"

llm = OpenAI(temperature=0)


loader = TextLoader('the_needed_text.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
text_store = Chroma.from_documents(texts, embeddings, collection_name="the_needed_text")

loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")
docs = loader.load()
ruff_texts = text_splitter.split_documents(docs)
ruff_store = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")

vectorstore_info = VectorStoreInfo(
    name="the_needed_text_in_detail",
    description="the most recent data of bill gates",
    vectorstore=text_store
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)


agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

agent_executor.run("who is bill gates?, what is his age now? and how many assets he has now? ")