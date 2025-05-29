#!/usr/bin/env python3
"""
rag_chat_ui.py
──────────────
Semantic-Kernel chat UI **with RAG + memory**:

1. Embeds every user query with Azure OpenAI.
2. Runs a vector search over Azure AI Search (`index01`, field = contentVector).
3. Feeds the top-k chunks into the chat prompt so answers are grounded.
4. Keeps full ChatHistory, so follow-up questions have conversational context.

Prereqs
• `pip install semantic-kernel azure-search-documents openai python-dotenv tqdm`
• The index schema must have: id, raw (Edm.String, searchable), contentVector (1536-d).
• `.env` contains the usual AZURE_* keys plus TEXT_EMBEDDING and CHAT deployments.

Usage
$ python rag_chat_ui.py
"""
import asyncio, os, sys, textwrap
from dotenv import load_dotenv
from tqdm.auto import tqdm

from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
)
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# ── env ────────────────────────────────────────────────────────
load_dotenv()

EMBED_DEPLOY = os.getenv("AZURE_TEXT_EMBEDDING_DEPLOYMENT_NAME")
CHAT_DEPLOY  = os.getenv("AZURE_OPENAI_REASONING_DEPLOYMENT_NAME",
                         os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
AOAI_BASE    = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
AOAI_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AOAI_VER     = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

VECTOR_FIELD = "contentVector"
TOP_K        = 3

# ── kernel & chat service ───────────────────────────────────────
kernel = Kernel()
chat_service = AzureChatCompletion(
    deployment_name=CHAT_DEPLOY,
    endpoint=AOAI_BASE,
    api_key=AOAI_KEY,
)
kernel.add_service(chat_service)
settings = AzureChatPromptExecutionSettings(max_tokens=400, temperature=0.3)

# ── search client ───────────────────────────────────────────────
search = SearchClient(
    endpoint   = os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "index01"),
    credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
    api_version="2024-07-01"
)

# ── embedding helper ────────────────────────────────────────────
embed_client = openai.AzureOpenAI(
    api_key     = AOAI_KEY,
    api_version = AOAI_VER,
    base_url    = f"{AOAI_BASE}/openai/deployments/{EMBED_DEPLOY}",
)

def embed(text: str) -> list[float]:
    return embed_client.embeddings.create(model=EMBED_DEPLOY, input=[text]).data[0].embedding

def retrieve_context(query: str, k: int = TOP_K) -> list[str]:
    vector = embed(query)
    vq = VectorizedQuery(vector=vector, fields=VECTOR_FIELD, k=k)
    hits = search.search(search_text="", vector_queries=[vq], top=k)
    return [doc["raw"] for doc in hits if "raw" in doc]

# ── main chat loop ──────────────────────────────────────────────
async def rag_chat() -> None:
    history = ChatHistory()
    print("🤖  RAG chat – ask me anything (exit to quit)")
    while True:
        try:
            user = input("🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        # 1) retrieve supporting context
        context_chunks = retrieve_context(user, TOP_K)
        context_block  = "\n\n".join(
            f"[Source {i+1}]\n{textwrap.shorten(c, 400)}"
            for i, c in enumerate(context_chunks)
        ) or "No relevant excerpts found."

        # 2) craft a system prompt with context
        system_prompt = (
            "You are an assistant answering questions with the help of report excerpts.\n"
            "Use them to ground your reply and cite [Source n] where appropriate.\n\n"
            f"Context:\n{context_block}"
        )

        history.add_system_message(system_prompt)
        history.add_user_message(user)

        reply = await chat_service.get_chat_message_content(
            history, settings, kernel=kernel
        )
        print("🤖 >", reply.content)

        history.add_assistant_message(reply.content)

        # remove the temporary system prompt so it doesn’t grow forever
        history.messages.pop(0)

if __name__ == "__main__":
    try:
        asyncio.run(rag_chat())
    except KeyboardInterrupt:
        print("\n👋  Bye!")
