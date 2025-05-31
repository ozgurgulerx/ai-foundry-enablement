# 250528-AI_Foundry
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)


# 🧠 AI Foundry Enablement

This repository contains advanced assets and enablement code for building GenAI-powered applications with Semantic Kernel, Azure Agent Service, and Retrieval-Augmented Generation (RAG). It is designed to accelerate internal prototyping, AI onboarding, and Foundry-style solution development.

> Build fast, iterate deeply, scale meaningfully.

---

## 📦 Key Modules

| File / Module                        | Description |
|--------------------------------------|-------------|
| `semantic_kernel/chat_ui.py`         | Basic chat interface built on Semantic Kernel |
| `semantic_kernel/plugins.py`         | Example plugin functions to extend capabilities |
| `semantic_kernel/stepwise_planner.py`| Stepwise task planner using SK's orchestration |
| `agent_service/base.py`              | Azure Agent Service integration base |
| `rag/baseline_rag.ipynb`             | Basic RAG pipeline in Jupyter |
| `rag/sk_rag.py`                      | RAG pipeline built on Semantic Kernel |
| `rag/agentic_rag_sk.py`              | Agentic RAG combining SK planning + retrieval |

---

## 🚀 Features

- 🔹 Semantic Kernel integration (chat, planner, plugins)
- 🔹 Baseline and agentic RAG workflows
- 🔹 Azure Agent Service scaffolding
- 🔹 Text extraction and dataset prep for IMF_WOO docs
- 🔹 Modular structure for experimentation and extension

---

## 📂 Folder Structure

```bash
ai-foundry-enablement/
│
├── agent_service/
│   └── base.py
│
├── semantic_kernel/
│   ├── chat_ui.py
│   ├── plugins.py
│   └── stepwise_planner.py
│
├── rag/
│   ├── baseline_rag.ipynb
│   ├── sk_rag.py
│   └── agentic_rag_sk.py
│
├── data/
│   ├── 2504_IMF_WOO.pdf
│   ├── 2504_IMF_WOO.txt
│   └── 2504_IMF_WOO.cleaned.txt
│
├── .env.template
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/ai-foundry-enablement.git
cd ai-foundry-enablement
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy the template and fill in your OpenAI / Azure credentials:

```bash
cp .env.template .env
```

`.env` example:

```env
OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=your-azure-key
```

---

## 🧠 Running Examples

### ✅ Run the Semantic Kernel Chat UI

```bash
python semantic_kernel/chat_ui.py
```

### ✅ Run the Baseline RAG Notebook

Open in Jupyter or VS Code:

```bash
jupyter notebook rag/baseline_rag.ipynb
```

### ✅ Run the Agentic RAG Script

```bash
python rag/agentic_rag_sk.py
```

---

## 🧠 IMF Dataset

The IMF World Outlook document is included in `.pdf`, `.txt`, and `.cleaned.txt` formats. These are used to:
- Generate embeddings
- Power RAG flows
- Demonstrate preprocessing

Use `txt` or `cleaned.txt` for ingest into vector stores.

---

## 📚 Requirements

- Python 3.10+
- `semantic-kernel`
- `openai`
- `langchain` (optional for vector stores)
- `dotenv`
- `pandas`, `tiktoken`, etc.

---

## 🔧 Requirements File

Here’s a sample from `requirements.txt`:

```txt
semantic-kernel==0.7.230628.1
openai==1.30.1
python-dotenv==1.0.1
tiktoken==0.5.1
pandas
```

---

## 🧱 Roadmap

- [ ] Enable vector DB plug-in for RAG
- [ ] Multi-agent orchestration via planner + tools
- [ ] Evaluation metrics & LLM eval integration

---

## 🛡 License

MIT License © Ozgur Guler

---

## 🙌 Acknowledgements

Built for AI onboarding, technical deep dives, and real-world prototyping. Inspired by:
- [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [Azure AI Studio](https://learn.microsoft.com/en-us/azure/ai-services/)


---

## ✨ Contact

For questions, reach out via [LinkedIn](https://linkedin.com/in/ozgurguler) or raise an issue.
