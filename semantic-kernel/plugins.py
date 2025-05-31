#!/usr/bin/env python3
"""
chat_with_logging.py – SK chat loop that logs every tool invocation
"""

import asyncio, os, logging
from datetime import datetime
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

# ── 1 · Logging config (console + timestamps) ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SK-Chat")

# ── 2 · Secrets ─────────────────────────────────────────────────────────
load_dotenv()

# ── 3 · Native tool (logs on each call) ─────────────────────────────────
class TimePlugin:
    @kernel_function(name="get_time", description="Return the current local time")
    def get_time(self) -> str:
        log.info("🔧  Function 'time.get_time' invoked by LLM")
        return f"🕒  It’s {datetime.now().strftime('%H:%M:%S')}."

# ── 4 · Kernel + service + plugin ───────────────────────────────────────
kernel = Kernel()

chat_service = AzureChatCompletion(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01-preview"),
)
kernel.add_service(chat_service)
kernel.add_plugin(TimePlugin(), plugin_name="time")

# enable auto function calls
settings = AzureChatPromptExecutionSettings()
settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

# ── 5 · Interactive chat loop with logging ─────────────────────────────
async def main() -> None:
    history = ChatHistory(system_message="You are a helpful assistant.")
    log.info("🟢  Chat session started. Ask for the time to trigger the tool.")

    while True:
        user = input("🧑 > ").strip()
        if user.lower() in {"exit", "quit"}:
            log.info("🔴  Session ended by user")
            break
        if not user:
            continue

        history.add_user_message(user)
        reply = await chat_service.get_chat_message_content(history, settings, kernel=kernel)
        print("🤖 >", reply.content)
        history.add_assistant_message(reply.content)

if __name__ == "__main__":
    asyncio.run(main())
