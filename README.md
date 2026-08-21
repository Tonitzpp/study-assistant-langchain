# Assistente de Estudos com LangChain / Study Assistant with LangChain

Chatbot de linha de comando que responde perguntas sobre anotações de estudo usando RAG + memória de conversa.

Command-line chatbot that answers questions about study notes using RAG + conversation memory.

---

## Tecnologias / Tech Stack

- LangChain + LangGraph
- Groq (GPT-OSS 20B)
- FAISS (busca vetorial / vector search)
- HuggingFace Embeddings (gratuito / free)
- Python 3.12+

---

## Como rodar / How to run

**1. Instale as dependências / Install dependencies**

```bash
pip install langchain langchain-groq langchain-community
pip install faiss-cpu sentence-transformers
```

**2. Crie o arquivo `.env` com sua chave / Create the `.env` file with your key**

```
GROQ_API_KEY=your_key_here
```

**3. Adicione seus arquivos de estudo / Add your study files**

Coloque seus arquivos `.md` ou `.txt` na pasta `documentos/`.

Place your `.md` or `.txt` files inside the `documentos/` folder.

**4. Rode o projeto / Run the project**

```bash
python main.py
```

---

## Como funciona / How it works

```
Você / You: o que é um PromptTemplate?
Bot: Com base nas suas anotações... / Based on your notes...

Você / You: e qual a diferença para o ChatPromptTemplate?
Bot: (lembra do contexto / remembers context)

Você / You: sair / exit
Bot: Até mais! / Goodbye!
```
