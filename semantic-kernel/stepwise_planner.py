# Check tomorrow’s weather in Rome; if it’s above 20 °C add ‘Morning run’ to my calendar, otherwise add ‘Gym session’; then tell me my next event.”
# Add 42 and 137, then give me a headline about AI that mentions that sum.
# Add 42 and 137, then give me a headline about AI that mentions that sum.”
#!/usr/bin/env python3
"""
planner_with_logging.py – Step-wise planning demo with 4 plugins + logging
(fixed for semantic-kernel >= 1.35-preview)
"""

import asyncio, logging, os, random
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.planners.function_calling_stepwise_planner import (
    FunctionCallingStepwisePlanner,
    FunctionCallingStepwisePlannerOptions,
)

# ── logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SK-Planner")

# ── secrets ────────────────────────────────────────────────────────────
load_dotenv()

# ── plugins (tools) ─────────────────────────────────────────────────────
class WeatherPlugin:
    @kernel_function(name="weather_today", description="Today's weather in °C")
    def weather_today(self, city: str) -> str:
        log.info(f"🔧 weather_today(city='{city}')")
        t = 15 + random.randint(-4, 4)
        return f"{city} is {t}°C and sunny."

    @kernel_function(name="weather_tomorrow", description="Tomorrow's weather in °C")
    def weather_tomorrow(self, city: str) -> str:
        log.info(f"🔧 weather_tomorrow(city='{city}')")
        t = 16 + random.randint(-4, 4)
        return f"Tomorrow {city} will be {t}°C with light clouds."


class CalendarPlugin:
    _events: Dict[str, str] = {}

    @kernel_function(name="add_event", description="Add event to calendar")
    def add_event(self, title: str, date: str) -> str:
        log.info(f"🔧 add_event(title='{title}', date='{date}')")
        self._events[date] = title
        return f"Event '{title}' added on {date}."

    @kernel_function(name="next_event", description="Show next calendar entry")
    def next_event(self) -> str:
        log.info("🔧 next_event()")
        if not self._events:
            return "Calendar is empty."
        d = min(self._events)
        return f"{d}: {self._events[d]}"


class MathPlugin:
    @kernel_function(name="add_numbers", description="Add two numbers")
    def add_numbers(self, a: float, b: float) -> str:
        log.info(f"🔧 add_numbers(a={a}, b={b})")
        return str(a + b)


class NewsPlugin:
    @kernel_function(name="headline", description="Give a headline about a topic")
    def headline(self, topic: str) -> str:
        log.info(f"🔧 headline(topic='{topic}')")
        return random.choice(
            [
                f"{topic} breakthrough shocks analysts",
                f"Experts react to latest {topic} update",
                f"How {topic} is reshaping the market",
            ]
        )

# ── kernel & service ───────────────────────────────────────────────────
kernel = Kernel()
chat_service = AzureChatCompletion(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01-preview"),
)
kernel.add_service(chat_service)

kernel.add_plugin(WeatherPlugin(), "weather")
kernel.add_plugin(CalendarPlugin(), "calendar")
kernel.add_plugin(MathPlugin(), "math")
kernel.add_plugin(NewsPlugin(), "news")

# ── planner ─────────────────────────────────────────────────────────────
planner_opts = FunctionCallingStepwisePlannerOptions(max_iterations=8)
planner = FunctionCallingStepwisePlanner(
    service_id=chat_service.service_id,
    options=planner_opts,
)

# ── interactive goal loop ───────────────────────────────────────────────
async def main() -> None:
    log.info(
        "🟢 Planner ready. Enter a GOAL (e.g. "
        "'Check tomorrow’s weather in Rome and add it to my calendar')."
    )
    while True:
        goal = input("🎯 > ").strip()
        if goal.lower() in {"exit", "quit"}:
            log.info("🔴 session ended")
            break
        if not goal:
            continue

        result = await planner.invoke(kernel, goal)
        log.info("✅ plan finished")

        print("\n📋 Scratch-pad (assistant messages):")
        for msg in result.chat_history.messages:
            if msg.role == "assistant":
                print("  •", msg.content)

        print("💡 Final answer:", result.final_answer)
        print("🔄 Iterations :", result.iterations, "\n")

if __name__ == "__main__":
    asyncio.run(main())
