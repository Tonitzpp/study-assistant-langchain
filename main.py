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
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import glob

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0.3,
    api_key = api_key
)
# model test
#print(model.invoke("olá").content)

files = (
    glob.glob("documentos/*.md") + 
    glob.glob("documentos/*.txt") +
    glob.glob("documentos/**/*.md", recursive=True) +
    glob.glob("documentos/**/.txt", recursive=True)
)
documents = sum([TextLoader(f, encoding = "utf-8").load() for f in files], [])
# loaded pages test
#print(f"{len(documents)} loaded pages")

chunks = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 100
).split_documents(documents)
# generated parts test
#print(f"{len(parts)} generated parts")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
recovered_data = FAISS.from_documents(
    chunks, embeddings
).as_retriever(search_kwargs = {"k": 3})

# embeddings and retriever test
#snippets = recovered_data.invoke("O que é Pandas?")
#print(snippets[0].page_content)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de respostas que responde as perguntas com base nos arquivos de anotações fornecidas."),
    ("placeholder", "{chat_history}"),
    ("human", "\n{query}\n\nAnotações relevantes: \n{context}")
])

