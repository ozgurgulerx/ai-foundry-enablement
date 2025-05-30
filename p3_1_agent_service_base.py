import os 
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import CodeInterpreterTool  # optional

load_dotenv()                               # pulls in PROJECT_ENDPOINT, etc.

endpoint   = os.getenv("AZURE_OPENAI_PROJECT_ENDPOINT")         # fail fast if missing
deployment = "gpt-4.1"

project_client = AIProjectClient(
    endpoint   = endpoint,
    credential = DefaultAzureCredential(),            # NOT the conn-string
    api_version = "2025-05-01",               # or "latest"
)

# ── 3. agent ───────────────────────────────────────────────────────────────
tool = CodeInterpreterTool()
agent = project_client.agents.create_agent(
    model        = deployment,
    name         = "my-agent",
    instructions = "You are a helpful assistant.",
    tools        = tool.definitions,                # drop this arg if you skipped the import
)
print("Agent:", agent.id)

# ── 4. conversation ────────────────────────────────────────────────────────
thread = project_client.agents.threads.create()
project_client.agents.messages.create(
    thread_id = thread.id,
    role      = "user",
    content   = "What is a blackhole?"
)

run = project_client.agents.runs.create_and_process(thread_id=thread.id,
                                                    agent_id=agent.id)
print("Run status:", run.status)

# ── 5. fetch replies ───────────────────────────────────────────────────────
for msg in project_client.agents.messages.list(thread_id=thread.id):
    # msg.content is a list of parts; for plain-text replies there’s one TextContentPart
    if msg.content and hasattr(msg.content[0], "text"):
        print(f"{msg.role}: {msg.content[0].text.value}")
    else:
        print(f"{msg.role}: [no text]")

# ── 6. cleanup (optional) ──────────────────────────────────────────────────
# project_client.agents.delete_agent(agent.id)