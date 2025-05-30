#!/usr/bin/env python3
"""
rag_chat_ui.py  –  SK chat with RAG + memory + logging  (API-2025-04-01)

Flow
────
1. Embed each user query (text-embedding-3-small).
2. Hit Azure AI Search (index01) – log every vector-DB round-trip.
3. Feed the top-k chunks into a chat-capable Azure OpenAI model.
4. Keep ChatHistory so follow-ups have context.

Patch history
• Uses max_completion_tokens (new API) and default temperature.
• Removes SDK warning by using top=.
• Adds INFO-level logging before and after each vector search.
"""

import asyncio, os, textwrap, openai, logging
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
)
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# ── logging setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RAG-Chat")

# ── env vars ─────────────────────────────────────────────────────
load_dotenv()

EMBED_DEPLOY = os.getenv("AZURE_TEXT_EMBEDDING_DEPLOYMENT_NAME")
CHAT_DEPLOY  = os.getenv("AZURE_OPENAI_REASONING_DEPLOYMENT_NAME",
                         os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
AOAI_BASE    = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
AOAI_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AOAI_VER     = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY      = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME      = os.getenv("AZURE_SEARCH_INDEX_NAME", "index01")

VECTOR_FIELD = "contentVector"
TOP_K        = 3

# ── SK kernel + chat service ─────────────────────────────────────
kernel = Kernel()
chat_service = AzureChatCompletion(
    deployment_name=CHAT_DEPLOY,
    endpoint=AOAI_BASE,
    api_key=AOAI_KEY,
)
kernel.add_service(chat_service)

settings = AzureChatPromptExecutionSettings(
    max_tokens=None,                               # disable old param
    additional_kwargs={"max_completion_tokens": 400}  # new param name
)

# ── Azure Search & embedding client ──────────────────────────────
search = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY),
    api_version="2024-07-01",
)

embed_client = openai.AzureOpenAI(
    api_key=AOAI_KEY,
    api_version=AOAI_VER,
    base_url=f"{AOAI_BASE}/openai/deployments/{EMBED_DEPLOY}",
)

def embed(text: str) -> list[float]:
    return embed_client.embeddings.create(model=EMBED_DEPLOY, input=[text]).data[0].embedding

def retrieve_context(query: str, k: int = TOP_K) -> list[str]:
    """Embed the query, hit vector DB, return up to k raw chunks."""
    log.info("🔍  Vector search for query: %s", query)
    vq = VectorizedQuery(vector=embed(query), fields=VECTOR_FIELD)
    hits = list(search.search(search_text="", vector_queries=[vq], top=k))
    log.info("✅  Retrieved %d chunk(s)", len(hits))
    return [doc["raw"] for doc in hits if "raw" in doc]

# ── chat loop ────────────────────────────────────────────────────
async def rag_chat() -> None:
    history = ChatHistory()
    print("🤖  RAG chat – ask me anything (type 'exit' to quit)")
    while True:
        try:
            user = input("🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        # vector retrieval
        excerpts = retrieve_context(user, TOP_K)
        context_block = "\n\n".join(
            f"[Source {i+1}]\n{textwrap.shorten(txt, 400)}"
            for i, txt in enumerate(excerpts)
        ) or "No relevant excerpts found."

        system_msg = (
            "You are an assistant who answers using the excerpts below.\n"
            "Ground every reply in these sources and cite [Source n] when relevant.\n\n"
            f"Excerpts:\n{context_block}"
        )

        history.add_system_message(system_msg)
        history.add_user_message(user)

        reply = await chat_service.get_chat_message_content(
            history, settings, kernel=kernel
        )
        print("🤖 ", reply.content, "\n")

        history.add_assistant_message(reply.content)
        history.messages.pop(0)   # remove temp system message per turn

if __name__ == "__main__":
    try:
        asyncio.run(rag_chat())
    except KeyboardInterrupt:
        print("\n👋  Bye!")
