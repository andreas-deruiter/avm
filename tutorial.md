# Tutorial: Building AVM as an MCP Server **with a Generative AI Meeting Scheduler Agent (Azure AI Foundry, GPT‑5, dotenv, VS Code)**

This end-to-end tutorial builds an **Agent–ViewModel (AVM)** server (HTTP + WebSocket; MCP‑aligned) **and** a **separate agent process** powered by **Azure AI Foundry** using **GPT‑5**. You’ll use **dotenv** for configuration and follow **VS Code** steps to set up and run everything locally. The agent uses LLM **tool calling** to: 1) read participants’ availability, 2) reason about options, 3) propose a slot, and 4) confirm the meeting. You’ll see **intents** (tools), **projections** (resources), and **multi‑client binding** in action.

> AVM ↔ MCP mapping: **Intents** ≈ Tools (HTTP endpoints) • **Projections** ≈ Resources (WebSocket subscriptions).

---

## 1) Prerequisites

* **Python** 3.10+
* **VS Code** with the **Python** extension
* An **Azure OpenAI (Azure AI Foundry)** resource with a **GPT‑5** chat deployment (name e.g. `gpt-5`).

### VS Code setup (recommended)
This part prepares a clean Python environment for the project and lists the exact packages your app needs. The virtual environment keeps dependencies isolated from anything else on your machine. The ```requirements.txt``` file is how you reproducibly install everything later on a teammate’s laptop or in CI/CD.

It’s needed so everyone runs the same versions of FastAPI, Pydantic, the Azure SDKs, etc., which prevents “works on my machine” bugs. VS Code’s interpreter selection makes sure the editor runs and debugs with your ```.venv.```

1. Open your project folder in VS Code.
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```
3. Select the interpreter: press `Ctrl/Cmd+Shift+P` → **Python: Select Interpreter** → choose **.venv**.
4. Create **requirements.txt**:

   ```txt
   fastapi
   uvicorn
   pydantic
   httpx
   websockets
   python-dateutil
   openai
   python-dotenv
   azure-ai-agents
   azure-identity
   ```
5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
6. Create a **.env** file (see below) and ensure VS Code’s terminal loads it (we call `dotenv.load_dotenv()` in code).

## .env configuration

The ```.env``` file stores secrets and runtime settings such as Azure endpoints, API keys, and sensible defaults for the agent. The code loads this file at startup so your credentials aren’t hard-coded.

It’s needed to keep secrets out of source control and to make switching environments easy. You can point at a different Azure project or model by changing environment variables, not code.

### .env example

Create `.env` in your project root:

```env
# Azure AI Foundry (Azure OpenAI)
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_API_VERSION=2024-07-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-5
# Azure AI Foundry Agents SDK project endpoint
AZURE_AI_PROJECT_ENDPOINT=https://<your-ai-foundry-project-endpoint>

# Agent defaults
ORGANIZER=organizer@example.com
PARTICIPANTS=alice@example.com,bob@example.com
WINDOW_START=2025-09-03T09:00:00
WINDOW_END=2025-09-03T17:00:00
TITLE=Design Sync

# Optional AVM base URLs
AVM_HTTP=http://localhost:8000
AVM_WS=ws://localhost:8000
```

> Tip: add `.env` to `.gitignore`.

---

## 2) Project Layout

```
avm_scheduler/
  ├── models.py          # Pydantic models for intents & projections
  ├── services.py        # Adapters: CalendarService (mock), MeetingService
  ├── avm.py             # Agent–ViewModel: intents, policy, projections
  ├── server.py          # FastAPI app exposing intents + subscriptions
  ├── agent_llm.py       # Generative AI agent (Azure OpenAI SDK, GPT‑5, dotenv)
  ├── agent_llm_agentsdk.py # Generative AI agent (Azure AI Foundry Agents SDK)
  └── supervisor.py      # Optional: approves/observes via projections
```

---

## 3) Models (models.py)
This module defines all the structured data that crosses your AVM boundary. Pydantic models like ```FreeBusyIn```, ```ProposeMeetingIn```, and ```MeetingProjection``` describe the shape of requests, responses, and streamed projection updates. ```CommandResult``` is a consistent envelope for success or error outcomes.

It’s needed to validate inputs automatically, generate clear errors for bad data, and make the API predictable for both humans and LLM agents. Strong typing here prevents a whole class of runtime bugs and miscommunications.

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List

# ---- Common ----
class CommandResult(BaseModel):
    ok: bool
    correlation_id: Optional[str] = None
    error: Optional[Literal["precondition_failed","conflict","requires_approval","forbidden","not_found"]] = None

class ProjectionEvent(BaseModel):
    projection: str
    version: int
    diff: Dict[str, Any]

# ---- Calendar / FreeBusy ----
class BusyBlock(BaseModel):
    start_iso: str
    end_iso: str

class FreeBusyIn(BaseModel):
    participants: List[str]
    window_start_iso: str
    window_end_iso: str

class FreeBusyOut(BaseModel):
    busy: Dict[str, List[BusyBlock]]

# ---- Meetings ----
class Slot(BaseModel):
    start_iso: str
    end_iso: str

class ProposeMeetingIn(BaseModel):
    organizer: str
    participants: List[str]
    slot: Slot
    title: str = Field(default="Meeting")
    idem_key: str

class ConfirmMeetingIn(BaseModel):
    meeting_id: str
    idem_key: str

class MeetingProjection(BaseModel):
    meeting_id: str
    status: Literal["Proposed","Confirmed","Cancelled"]
    organizer: str
    participants: List[str]
    slot: Slot
    approvals: Dict[str, Literal["Pending","Approved","Rejected"]] = {}
```

---

## 4) Services / Adapters (services.py)
These are adapters that mimic real backends. ```CalendarService``` exposes simple “free/busy” reads and lets you add events. ```MeetingService``` stores meetings and their status. In production these would call Microsoft Graph, Google Calendar, or a database; here they’re in-memory so you can run the whole flow locally.

They’re needed because AVM intentionally hides backend complexity from agents. By coding against adapters, you can swap implementations later without touching the ViewModel or the agent.

```python
# services.py
from __future__ import annotations
from typing import Dict, List, Tuple
from dateutil.parser import isoparse

class CalendarService:
    def __init__(self):
        self._busy: Dict[str, List[Tuple[str,str]]] = {
            "alice@example.com": [("2025-09-03T09:30:00","2025-09-03T10:30:00")],
            "bob@example.com":   [("2025-09-03T11:00:00","2025-09-03T12:00:00")],
            "carol@example.com": [("2025-09-03T09:00:00","2025-09-03T09:30:00")],
        }

    def freebusy(self, participants: List[str], start_iso: str, end_iso: str) -> Dict[str, List[Tuple[str,str]]]:
        start = isoparse(start_iso); end = isoparse(end_iso)
        out: Dict[str, List[Tuple[str,str]]] = {}
        for p in participants:
            blocks = []
            for s,e in self._busy.get(p, []):
                if isoparse(e) > start and isoparse(s) < end:
                    blocks.append((s,e))
            out[p] = blocks
        return out

    def create_event(self, participants: List[str], start_iso: str, end_iso: str, title: str) -> None:
        for p in participants:
            self._busy.setdefault(p, []).append((start_iso, end_iso))

class MeetingService:
    def __init__(self):
        self._meetings: Dict[str, Dict] = {}
        self._counter = 0

    def new_meeting_id(self) -> str:
        self._counter += 1
        return f"M-{self._counter}"

    def save(self, meeting: Dict) -> None:
        self._meetings[meeting["meeting_id"]] = meeting

    def get(self, meeting_id: str) -> Dict:
        return self._meetings[meeting_id]

    def confirm(self, meeting_id: str) -> None:
        self._meetings[meeting_id]["status"] = "Confirmed"
```

---

## 5) AVM Core (avm.py)
This is the heart of the pattern. ```ProjectionStore``` is a tiny in-memory pub/sub that tracks a projection’s current snapshot and version and pushes diffs to subscribers. ```AgentViewModel``` exposes high-level intents such as ```calendar_freebusy```, ```meeting_propose```, and ```meeting_confirm```. When an intent changes state, the ViewModel publishes a projection update so all clients (agents, UIs, supervisors) stay in sync.

It’s needed to enforce business rules at the boundary (for example, conflict checking before proposing a meeting), to keep agents on least-privilege rails (only the shaped data they need), and to support multi-client real-time binding without each client talking to backends directly.

```python
# avm.py
from __future__ import annotations
import asyncio
from typing import Dict, Any, List
from models import (
    ProjectionEvent, CommandResult,
    FreeBusyIn, FreeBusyOut, BusyBlock,
    ProposeMeetingIn, ConfirmMeetingIn
)
from services import CalendarService, MeetingService

class Projection:
    def __init__(self, projection_id: str):
        self.id = projection_id
        self.version = 0
        self.snapshot: Dict[str, Any] = {}
        self.subscribers: List[asyncio.Queue] = []

class ProjectionStore:
    def __init__(self):
        self._p: Dict[str, Projection] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, projection_id: str) -> asyncio.Queue:
        async with self._lock:
            proj = self._p.setdefault(projection_id, Projection(projection_id))
            q: asyncio.Queue = asyncio.Queue(maxsize=256)
            proj.subscribers.append(q)
            await q.put(ProjectionEvent(projection=proj.id, version=proj.version, diff=dict(proj.snapshot)))
            return q

    async def update(self, projection_id: str, diff: Dict[str, Any]) -> ProjectionEvent:
        async with self._lock:
            proj = self._p.setdefault(projection_id, Projection(projection_id))
            proj.version += 1
            proj.snapshot.update(diff)
            ev = ProjectionEvent(projection=proj.id, version=proj.version, diff=diff)
            for q in list(proj.subscribers):
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    proj.subscribers.remove(q)
            return ev

class AgentViewModel:
    def __init__(self, calendars: CalendarService, meetings: MeetingService, proj: ProjectionStore):
        self.cal = calendars
        self.mt = meetings
        self.proj = proj

    async def calendar_freebusy(self, args: FreeBusyIn) -> FreeBusyOut:
        busy = self.cal.freebusy(args.participants, args.window_start_iso, args.window_end_iso)
        shaped = {p: [BusyBlock(start_iso=s, end_iso=e) for (s,e) in blocks] for p,blocks in busy.items()}
        return FreeBusyOut(busy=shaped)

    async def meeting_propose(self, a: ProposeMeetingIn) -> CommandResult:
        fb = self.cal.freebusy(a.participants + [a.organizer], a.slot.start_iso, a.slot.end_iso)
        for _, blocks in fb.items():
            if blocks:
                return CommandResult(ok=False, error="conflict")
        meeting_id = self.mt.new_meeting_id()
        meeting = {
            "meeting_id": meeting_id,
            "status": "Proposed",
            "organizer": a.organizer,
            "participants": a.participants,
            "slot": a.slot.model_dump(),
            "approvals": {u: "Pending" for u in a.participants}
        }
        self.mt.save(meeting)
        await self.proj.update(f"meeting:{meeting_id}", meeting)
        return CommandResult(ok=True, correlation_id=meeting_id)

    async def meeting_confirm(self, a: ConfirmMeetingIn) -> CommandResult:
        m = self.mt.get(a.meeting_id)
        slot = m["slot"]
        self.cal.create_event([m["organizer"], *m["participants"]], slot["start_iso"], slot["end_iso"], title="Meeting")
        self.mt.confirm(a.meeting_id)
        await self.proj.update(f"meeting:{a.meeting_id}", {"status": "Confirmed"})
        return CommandResult(ok=True, correlation_id=a.meeting_id)
```

---

## 6) FastAPI Server (server.py)
This file turns the AVM into a network service. It exposes each intent as a POST endpoint under ```/intents/*```, and it exposes projections over WebSockets at ```/subscriptions/{projection_id}```. When a client subscribes, it immediately receives the current snapshot, then live diffs as they happen.

It’s needed so your agent can act via HTTP and react via WebSockets. This is also what lets multiple clients subscribe to the same projection and see consistent state without polling.

```python
# server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from models import FreeBusyIn, FreeBusyOut, ProposeMeetingIn, ConfirmMeetingIn, CommandResult
from avm import ProjectionStore, AgentViewModel
from services import CalendarService, MeetingService

app = FastAPI(title="AVM – Meeting Scheduler")
proj = ProjectionStore()
cal = CalendarService()
mt = MeetingService()
avm = AgentViewModel(cal, mt, proj)

@app.post("/intents/calendar.freebusy", response_model=FreeBusyOut)
async def calendar_freebusy(args: FreeBusyIn):
    return await avm.calendar_freebusy(args)

@app.post("/intents/meeting.propose", response_model=CommandResult)
async def meeting_propose(args: ProposeMeetingIn):
    return await avm.meeting_propose(args)

@app.post("/intents/meeting.confirm", response_model=CommandResult)
async def meeting_confirm(args: ConfirmMeetingIn):
    return await avm.meeting_confirm(args)

@app.websocket("/subscriptions/{projection_id}")
async def subscribe(ws: WebSocket, projection_id: str):
    await ws.accept()
    q = await proj.subscribe(projection_id)
    try:
        while True:
            ev = await q.get()
            await ws.send_json(ev.model_dump())
    except WebSocketDisconnect:
        return
```

Run the server (VS Code terminal with .venv active):

```bash
uvicorn server:app --reload
```

---

## 7) Generative AI Agent (agent\_llm\_agentsdk.py) — GPT‑5 + dotenv + **Azure AI Foundry Agents SDK**

This is the generative agent that runs as a separate process. It defines three function tools that call your AVM intents over HTTP. With the Agents SDK, the LLM can automatically decide to call those tools, reason over the returned data, and then continue the conversation. The script seeds a conversation with scheduling context, streams the run, and starts listening to the meeting’s projection once it’s created.

It’s needed to connect LLM reasoning to safe, auditable actions. The agent never touches backends; it only calls the typed tools you exposed. That separation makes it easier to review, log, and test actions while still getting the benefits of GPT-5’s planning and language skills.

Below is a version of the agent implemented with the **Azure AI Foundry Agents SDK** (the new agent framework). It registers your AVM intents as **function tools**, lets the SDK **auto‑invoke** them, and streams results.

> One‑time install (VS Code terminal): `pip install azure-ai-agents azure-identity`

Add this file as `agent_llm_agentsdk.py`:

```python
# agent_llm_agentsdk.py
import os, json, uuid, asyncio, httpx, websockets
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.tool import AsyncFunctionTool, AsyncToolSet
from azure.ai.agents.models import MessageRole

load_dotenv()  # loads .env

# --- AVM endpoints ---
AVM_HTTP = os.getenv("AVM_HTTP", "http://localhost:8000")
AVM_WS   = os.getenv("AVM_WS",   "ws://localhost:8000")

# --- Azure AI Foundry Agents SDK ---
# Use your Azure AI Foundry **project endpoint** (Project > Overview > Endpoint)
AZURE_AI_PROJECT_ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
MODEL = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5")

credential = DefaultAzureCredential()

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

# ---- Register tools with schemas (auto function calling) ----
functions = AsyncFunctionTool([
    {
        "name": "calendar_freebusy",
        "description": "Get busy intervals for participants within a time window.",
        "parameters": {
            "type": "object",
            "properties": {
                "participants": {"type": "array", "items": {"type": "string"}},
                "window_start_iso": {"type": "string"},
                "window_end_iso": {"type": "string"}
            },
            "required": ["participants", "window_start_iso", "window_end_iso"]
        },
        "code": tool_calendar_freebusy,
    },
    {
        "name": "propose_meeting",
        "description": "Propose a meeting slot and return meeting_id on success.",
        "parameters": {
            "type": "object",
            "properties": {
                "organizer": {"type": "string"},
                "participants": {"type": "array", "items": {"type": "string"}},
                "start_iso": {"type": "string"},
                "end_iso": {"type": "string"},
                "title": {"type": "string"}
            },
            "required": ["organizer", "participants", "start_iso", "end_iso"]
        },
        "code": tool_propose_meeting,
    },
    {
        "name": "confirm_meeting",
        "description": "Confirm a previously proposed meeting.",
        "parameters": {
            "type": "object",
            "properties": {"meeting_id": {"type": "string"}},
            "required": ["meeting_id"]
        },
        "code": tool_confirm_meeting,
    },
])

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
    agents = AgentsClient(endpoint=AZURE_AI_PROJECT_ENDPOINT, credential=credential)
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
    meeting_id = None
    async with agents.runs.stream(thread_id=thread.id, agent_id=agent.id) as stream:
        async for event_type, event_data, _ in stream:
            print(event_type, getattr(event_data, "status", ""))
            if hasattr(event_data, "text"):
                print("[agent]", getattr(event_data, "text", None))
            if hasattr(event_data, "output") and isinstance(event_data.output, dict):
                mid = event_data.output.get("correlation_id")
                if mid and not meeting_id:
                    meeting_id = mid
                    asyncio.create_task(subscribe_meeting(meeting_id))

if __name__ == "__main__":
    asyncio.run(main())
```

Run (VS Code terminal, `.venv` active):

```bash
python agent_llm_agentsdk.py
```

If the SDK is not auto‑calling functions in your environment, switch to `runs.create_and_process(...)` instead of streaming, or verify your `.env` contains `AZURE_AI_PROJECT_ENDPOINT` and your identity has access to the Azure AI Foundry project.

---

## 8) Optional Supervisor (supervisor.py)
This tiny script opens a WebSocket to a meeting projection and prints every update. It behaves like a simple “second client,” which could be a human dashboard or another automation.

It’s needed to demonstrate multi-client binding. Because projections are broadcast, any number of clients can watch the same state and act accordingly, which is useful for human-in-the-loop approval flows or monitoring.

```python
# supervisor.py
import asyncio, json, websockets

WS = "ws://localhost:8000"

async def watch(meeting_id: str):
    uri = f"{WS}/subscriptions/meeting:{meeting_id}"
    async with websockets.connect(uri) as ws:
        async for msg in ws:
            print("[supervisor]", json.loads(msg))

if __name__ == "__main__":
    asyncio.run(watch("meeting:M-1"))
```

---

## 9) End-to-End Test (VS Code workflow)
The test steps start the AVM server and the agent in separate terminals. You should see the agent call ```calendar.freebusy```, propose a slot, subscribe to the projection, and confirm the meeting. The ```launch.json``` snippet lets VS Code run the server with your ```.env``` loaded.

It’s needed to verify the full round-trip: intent calls, projection updates, and the agent’s tool-driven actions. Once this works locally, you can replace the mock services with real adapters and keep the rest unchanged.

1. Open folder → ensure **.venv** interpreter is selected.
2. `uvicorn server:app --reload` to start AVM server.
3. New terminal → run either agent:

   * **Azure OpenAI SDK** variant: `python agent_llm.py`
   * **Azure AI Foundry Agents SDK** variant: `python agent_llm_agentsdk.py`
4. Watch logs: tool calls to `calendar.freebusy` → `meeting.propose` → projection updates → `meeting.confirm`.

(Optional) Create a **launch.json** for server:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "AVM Server",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["server:app", "--reload"],
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}
```

---

## 10) Production Notes & Next Steps

* **Boundary safety**: LLM acts only via typed AVM tools.
* **Multi‑client**: Agents and UIs subscribe to the same projections.
* **dotenv**: Centralizes env config; pair with secret storage in production.
* Add **two‑phase participant approvals** and per‑client scopes/rate limits.
* Use **durable pub/sub** (Redis/NATS) behind the Projection Store for scale.
* Integrate Microsoft Graph/Google Calendar adapters and optimistic concurrency.

You now have a realistic AVM server and a **GPT‑5 scheduling agent** using either the classic Azure OpenAI SDK or the **Azure AI Foundry Agents SDK**, configured via **dotenv**, runnable end‑to‑end in **VS Code**.
