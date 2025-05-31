#!/usr/bin/env python3

"""
This script sets up a minimal interactive chat loop using Semantic Kernel with Azure OpenAI. 
It loads credentials from a .env file, builds a Kernel, registers an AzureChatCompletion service, 
and initializes a ChatHistory to retain conversation context. 
In a loop, it takes user input, sends it to the model using get_chat_message_content, 
prints the assistant's response, and appends it to the history, enabling continuous, contextual conversation. 
The chat ends cleanly on exit, quit, or interruption.
"""


import asyncio, os, sys
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
)

# ── 1. env vars ─────────────────────────────────────────────────────────
load_dotenv()                                                   # .env with AZURE_* vars
DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

# ── 2. kernel + service ─────────────────────────────────────────────────
kernel = Kernel()
chat_service = AzureChatCompletion(
    deployment_name=DEPLOYMENT,
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
kernel.add_service(chat_service)

settings = AzureChatPromptExecutionSettings(temperature=0.7, max_tokens=400)

# ── 3. interactive loop ─────────────────────────────────────────────────
async def chat_loop() -> None:
    history = ChatHistory()                                     # keeps the full convo
    print("🤖  SK chat – type 'exit' to quit")
    while True:
        try:
            user = input("🧑 > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        history.add_user_message(user)

        reply = await chat_service.get_chat_message_content(history, settings, kernel=kernel)
        print("🤖 >", reply.content)
        history.add_assistant_message(reply.content)

if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\n👋  Bye!")
