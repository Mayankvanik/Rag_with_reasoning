Got it 👍 Here’s a clean **README.md** draft for your project:

````markdown
# RAG with Reasoning

This project demonstrates a **RAG (Retrieval-Augmented Generation) system with reasoning capabilities**.  
It uses:
- **ChromaDB** as the vector database
- **MongoDB** to store chat history and metadata
- **OpenAI** as the LLM and embedding model

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Mayankvanik/Rag_with_reasoning.git
cd Rag_with_reasoning
````

### 2. Install `uv`

```bash
pip install uv
```

### 3. Create a virtual environment

```bash
uv venv
```

### 4. Install dependencies

```bash
uv sync
```

### 5. Setup environment variables

* Create a `.env` file in the project root.
* Copy values from the provided `sample.env` and add your credentials.

### 6. Run the server

```bash
uv run -m app.main
```

---

## 📦 Tech Stack

* **Vector DB**: [ChromaDB](https://www.trychroma.com/)
* **Database**: MongoDB
* **LLM & Embeddings**: OpenAI

---

## 📑 Notes

* Ensure MongoDB and ChromaDB are running before starting the server.
* You must have a valid OpenAI API key in your `.env`.

```

Do you want me to also add **usage examples** (like how to query the chatbot) or just keep it minimal for setup only?
```
