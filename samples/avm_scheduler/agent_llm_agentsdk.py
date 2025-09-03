# agent_llm_agentsdk.py
import os, json, uuid, asyncio, httpx, websockets
from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import AsyncFunctionTool, AsyncToolSet, MessageRole

load_dotenv()  # loads .env

def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Set it in your .env or shell. See .env.sample for a template."
        )
    return val

# --- AVM endpoints ---
AVM_HTTP = os.getenv("AVM_HTTP", "http://localhost:8000")
AVM_WS   = os.getenv("AVM_WS",   "ws://localhost:8000")

# --- Azure AI Foundry Agents SDK ---
# Use your Azure AI Foundry **project endpoint** (Project > Overview > Endpoint)
AZURE_AI_PROJECT_ENDPOINT = require_env("AZURE_AI_PROJECT_ENDPOINT")
MODEL = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5")

credential = AsyncDefaultAzureCredential()

# --- Domain defaults from .env ---
ORGANIZER     = os.environ.get("ORGANIZER", "organizer@example.com")
PARTICIPANTS  = os.environ.get("PARTICIPANTS", "alice@example.com,bob@example.com").split(",")
WINDOW_START  = os.environ.get("WINDOW_START", "2025-09-03T09:00:00")
WINDOW_END    = os.environ.get("WINDOW_END",   "2025-09-03T17:00:00")
TITLE         = os.environ.get("TITLE", "Design Sync")

# ---- Tool implementations that call AVM intents ----
async def tool_calendar_freebusy(participants: list[str], window_start_iso: str, window_end_iso: str) -> dict:
    payload = {"participants": participants, "window_start_iso": window_start_iso, "window_end_iso": window_end_iso}
    async with httpx.AsyncClient() as http:
        r = await http.post(f"{AVM_HTTP}/intents/calendar.freebusy", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()

async def tool_propose_meeting(organizer: str, participants: list[str], start_iso: str, end_iso: str, title: str | None = None) -> dict:
    payload = {"organizer": organizer, "participants": participants, "slot": {"start_iso": start_iso, "end_iso": end_iso}, "title": title or TITLE, "idem_key": str(uuid.uuid4())}
    async with httpx.AsyncClient() as http:
        r = await http.post(f"{AVM_HTTP}/intents/meeting.propose", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()

async def tool_confirm_meeting(meeting_id: str) -> dict:
    payload = {"meeting_id": meeting_id, "idem_key": str(uuid.uuid4())}
    async with httpx.AsyncClient() as http:
        r = await http.post(f"{AVM_HTTP}/intents/meeting.confirm", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()

# ---- Register tools (auto function calling) ----
functions = AsyncFunctionTool({
    tool_calendar_freebusy,
    tool_propose_meeting,
    tool_confirm_meeting,
})

toolset = AsyncToolSet()
toolset.add(functions)

SYSTEM_INSTRUCTIONS = (
    "You are a careful scheduling agent. Find the earliest 30-minute slot where ALL participants are free "
    "between the provided window. Call tools to check free/busy, then propose the slot. If accepted, confirm the meeting. "
    "Be concise in messages."
)

async def subscribe_meeting(meeting_id: str):
    uri = f"{AVM_WS}/subscriptions/meeting:{meeting_id}"
    async with websockets.connect(uri) as ws:
        async for msg in ws:
            print("[projection]", json.loads(msg))

async def main():
    async with AgentsClient(endpoint=AZURE_AI_PROJECT_ENDPOINT, credential=credential) as agents:
        # Enable auto function calls so the SDK executes our functions when the model requests them
        agents.enable_auto_function_calls(toolset)

        # Create agent with tools attached
        agent = await agents.create_agent(
            model=MODEL,
            name="scheduler-agent",
            instructions=SYSTEM_INSTRUCTIONS,
            toolset=toolset,
        )

        # Create a conversation thread and seed it with scheduling context
        thread = await agents.threads.create()
        user_payload = {
            "organizer": ORGANIZER,
            "participants": PARTICIPANTS,
            "window_start_iso": WINDOW_START,
            "window_end_iso": WINDOW_END,
            "duration_minutes": 30,
            "title": TITLE,
        }
        await agents.messages.create(thread_id=thread.id, role=MessageRole.USER, content=json.dumps(user_payload))

        # Stream processing (SDK will invoke our tools automatically)
        stream_ctx = await agents.runs.stream(thread_id=thread.id, agent_id=agent.id)
        async with stream_ctx as stream:
            async for event_type, event_data, _ in stream:
                print(event_type, getattr(event_data, "status", ""))
                if hasattr(event_data, "text"):
                    print("[agent]", getattr(event_data, "text", None))

if __name__ == "__main__":
    asyncio.run(main())