# 🔍 AskMyDocs — AI Document Q&A with Gemma2 & Groq

> Upload your PDFs and ask anything — get instant, accurate answers powered by **Gemma2**, **Groq's ultra-fast inference**, and **RAG (Retrieval-Augmented Generation)**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-API-orange?style=flat-square)
![Google Gemini](https://img.shields.io/badge/Google-Gemini%20Embeddings-blue?style=flat-square&logo=google)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-purple?style=flat-square)

---

## 📌 What is This?

**AskMyDocs** is a Retrieval-Augmented Generation (RAG) app that lets you:

- 📄 Load any PDF documents from a local folder
- 🧩 Split and embed them into a FAISS vector store
- 💬 Ask natural language questions about your documents
- ⚡ Get lightning-fast answers via Groq's inference engine
- 📍 See exactly which document chunks were used to answer

No hallucinations. No guessing. Answers come **only** from your documents.

---

## 🚀 Demo

```
User: "What are the key findings of the research paper?"
AskMyDocs: "According to Chapter 3 (page 12), the key findings include..."
            ↳ Source: research_paper.pdf · page 12
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **LLM** | Gemma2-9b-it via Groq API |
| **Embeddings** | Google Generative AI (`embedding-001`) |
| **Vector Store** | FAISS (in-memory) |
| **RAG Framework** | LangChain |
| **Document Loader** | PyPDFDirectoryLoader |

---

## 📁 Project Structure

```
askmydocs/
│
├── app.py                  # Main Streamlit application
├── .env                    # API keys (never commit this!)
├── .env.example            # Template for environment variables
├── requirements.txt        # Python dependencies
├── README.md               # You are here
│
└── data/                   # 📂 Put your PDF files here
    ├── document1.pdf
    ├── document2.pdf
    └── ...
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/askmydocs.git
cd askmydocs
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

Get your keys here:
- 🔑 **Groq API Key** → [console.groq.com](https://console.groq.com)
- 🔑 **Google API Key** → [aistudio.google.com](https://aistudio.google.com)

### 5. Add Your PDFs

```bash
mkdir data
# Copy your PDF files into the data/ folder
cp your_documents/*.pdf data/
```

### 6. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📖 How to Use

```
Step 1 → Place your PDF files in the ./data folder
Step 2 → Open the app and click "Build / Rebuild Index" in the sidebar
Step 3 → Wait for the vector store to be ready (green status indicator)
Step 4 → Type your question and click "Ask"
Step 5 → View the answer + source chunks used
```

### Sidebar Settings

| Setting | Default | Description |
|---------|---------|-------------|
| PDF Directory | `./data` | Folder path containing your PDFs |
| Chunk Size | 1000 | Token size of each document chunk |
| Chunk Overlap | 200 | Overlap between adjacent chunks |
| Max Pages | 20 | Maximum pages to index (rate limit safety) |

---

## 📦 Requirements

```txt
streamlit>=1.32.0
langchain>=0.2.0
langchain-groq>=0.1.0
langchain-google-genai>=1.0.0
langchain-community>=0.2.0
faiss-cpu>=1.7.4
pypdf>=4.0.0
python-dotenv>=1.0.0
```

Install all at once:

```bash
pip install streamlit langchain langchain-groq langchain-google-genai langchain-community faiss-cpu pypdf python-dotenv
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Your Groq Cloud API key |
| `GOOGLE_API_KEY` | ✅ Yes | Your Google AI Studio API key |

> ⚠️ **Never commit your `.env` file to GitHub!** Add it to `.gitignore`.

```bash
# .gitignore
.env
__pycache__/
*.pyc
data/
venv/
```

---

## 🧠 How RAG Works (Under the Hood)

```
Your PDFs
    ↓
[PyPDFDirectoryLoader] — loads all PDF pages
    ↓
[RecursiveCharacterTextSplitter] — splits into chunks (1000 tokens, 200 overlap)
    ↓
[Google Embedding-001] — converts chunks into vectors
    ↓
[FAISS Vector Store] — stores vectors in memory
    ↓
User Question → [Similarity Search] → Top 5 relevant chunks
    ↓
[Gemma2-9b-it via Groq] — generates answer from chunks only
    ↓
Answer + Source Chunks displayed in UI
```

---

## ⚡ Why Groq?

Groq's LPU (Language Processing Unit) delivers **10x faster inference** than traditional GPU-based APIs. This means:

- Near-instant responses even for long documents
- Lower latency for real-time Q&A
- Cost-effective for high-volume usage

---

## 🐛 Troubleshooting

**No PDFs found error?**
```bash
# Make sure your data folder exists and has PDF files
ls data/*.pdf
```

**API Key error?**
```bash
# Check your .env file has correct keys with no spaces
cat .env
```

**Rate limit hit?**
- Reduce "Max Pages to Index" in sidebar (try 10-15)
- Wait a few seconds and try rebuilding the index

**FAISS install issues on Windows?**
```bash
pip install faiss-cpu --no-cache-dir
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

Built with ❤️ using LangChain, Groq, and Google Gemini Embeddings.

---

> 💡 **Pro Tip:** For best results, use well-structured PDFs with clear headings. Scanned image-based PDFs may not extract text correctly.
