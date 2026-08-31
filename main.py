import warnings
import os
import logging

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

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
    chunk_size = 1200,
    chunk_overlap = 200
).split_documents(documents)
# generated parts test
#print(f"{len(parts)} generated parts")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
recovered_data = FAISS.from_documents(
    chunks, embeddings
).as_retriever(search_kwargs = {"k": 6})

# embeddings and retriever test
#snippets = recovered_data.invoke("O que é Pandas?")
#print(snippets[0].page_content)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é o Yoshi, um assistente de estudos especializado.
    Responda as perguntas com base nas anotações fornecidas.
    Se a resposta não estiver explícita nas anotações, use o conteúdo disponível para raciocinar e chegar à resposta.
    Só diga que não sabe se o assunto for completamente diferente do que está nas anotações."""),
    ("placeholder", "{chat_history}"),
    ("human", "\n{query}\n\nAnotações relevantes: \n{context}")
])

chain = prompt | model | StrOutputParser()

memory = {}

# função para o histórico de sessões, se a sessão não existe ela é criada em memory
def get_session_history(session:str):
    if session not in memory:
        memory[session] = InMemoryChatMessageHistory()
    return memory[session]

chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history = get_session_history,
    input_messages_key="query",
    history_messages_key="chat_history"
)

session = "study_01"

print("\nOlá! Eu sou o Yoshi")
print("Estou aqui para responder suas dúvidas com base nas suas anotações de estudo.")
print('Digite "sair" para encerrar.\n')
while True:
    question = input("Você: ")
    if question.lower() == "sair":
        print("Até mais! :)")
        break

    snippets = recovered_data.invoke(question)
    context = "\n\n".join(s.page_content for s in snippets)

    awnser = chain_with_memory.invoke(
        {"query": question, "context": context},
        config={"configurable": {"session_id": session}}
    )
    print(f"\nYoshi: {awnser}")

