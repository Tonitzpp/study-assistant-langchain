from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import glob

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0.3,
    api_key = api_key
)

files = glob.blob("documentos/*.md") + glob.glob("documentos/*.txt")
documents = sum([TextLoader(f, encoding = "utf-8").load() for f in files], [])

parts = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 100
).split_documents(documents)

