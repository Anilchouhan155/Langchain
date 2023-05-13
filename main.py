import os
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA

os.environ['serpapi_api_key']="f25aae30b34fb5273f6c65c7bd4a46aa91a6ad25b941c1e217e183fdc8b16c55"
os.environ['OPENAI_API_KEY']="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"

llm = OpenAI(temperature=0)


from langchain.document_loaders import TextLoader
loader = TextLoader('the_needed_text.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
state_of_union_store = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")

from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")
docs = loader.load()
ruff_texts = text_splitter.split_documents(docs)
ruff_store = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

vectorstore_info = VectorStoreInfo(
    name="state_of_union_address",
    description="the most recent state of the Union adress",
    vectorstore=state_of_union_store
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

agent_executor.run("who is bill gates?, what is his age now? and how many assets he has now? ")