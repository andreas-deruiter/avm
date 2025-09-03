# Tutorial: Building AVM as an MCP Server with a Generative AI Meeting Scheduler Agent (Azure AI Foundry)

This end-to-end tutorial builds an **Agent–ViewModel (AVM)** server (HTTP + WebSocket; MCP‑aligned) **and** a **separate agent process** powered by **Azure AI Foundry (Azure OpenAI)**. The agent uses an LLM with **tool calling** to: 1) read participants’ availability, 2) reason about options, 3) propose a meeting slot, and 4) confirm the meeting. You’ll see **intents** (tools), **projections** (resources), **two‑phase readiness**, least‑privilege, and **multi‑client binding** in action.

> AVM ↔ MCP mapping: **Intents** ≈ Tools (HTTP endpoints) • **Projections** ≈ Resources (WebSocket subscriptions).

---

## 1) Prerequisites

* Python 3.10+
* An Azure OpenAI (Azure AI Foundry) deployment with a chat model (e.g., `gpt-4o`/`gpt-35-turbo` equivalent).
* Environment variables set:

  * `AZURE_OPENAI_ENDPOINT` (e.g., `https://<your-resource>.openai.azure.com`)
  * `AZURE_OPENAI_API_KEY`
  * `AZURE_OPENAI_API_VERSION` (e.g., `2024-07-01-preview`)
  * `AZURE_OPENAI_DEPLOYMENT` (your model deployment name)
* Install dependencies:

```bash
pip install fastapi uvicorn pydantic httpx websockets python-dateutil openai
```

---

## 2) Project Layout

```
avm_scheduler/
  ├── models.py          # Pydantic models for intents & projections
  ├── services.py        # Adapters: CalendarService (mock), MeetingService
  ├── avm.py             # Agent–ViewModel: intents, policy, projections
  ├── server.py          # FastAPI app exposing intents + subscriptions
  ├── agent_llm.py       # Generative AI agent (Azure AI Foundry)
  └── supervisor.py      # Optional: approves/observes via projections
```

> If you already built the earlier server files, you only need to add/replace the agent with `agent_llm.py` below.

---

## 3) Models (models.py)

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

Run the server:

```bash
uvicorn server:app --reload
```

---

## 7) Generative AI Agent (agent\_llm.py) — Azure AI Foundry

```python
# agent_llm.py
import os, json, asyncio, uuid
import httpx, websockets
from openai import AzureOpenAI

AVM_HTTP = "http://localhost:8000"
AVM_WS   = "ws://localhost:8000"

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-07-01-preview"),
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
MODEL = os.environ["AZURE_OPENAI_DEPLOYMENT"]

ORGANIZER = "organizer@example.com"
PARTICIPANTS = ["alice@example.com", "bob@example.com"]
WINDOW_START = "2025-09-03T09:00:00"
WINDOW_END   = "2025-09-03T17:00:00"
TITLE        = "Design Sync"

TOOLS = [
    {"type": "function", "function": {"name": "calendar_freebusy", "description": "Get busy intervals", "parameters": {"type": "object", "properties": {"participants": {"type": "array", "items": {"type": "string"}}, "window_start_iso": {"type": "string"}, "window_end_iso": {"type": "string"}}, "required": ["participants","window_start_iso","window_end_iso"]}}},
    {"type": "function", "function": {"name": "propose_meeting", "description": "Propose meeting slot", "parameters": {"type": "object", "properties": {"organizer": {"type": "string"}, "participants": {"type": "array", "items": {"type": "string"}}, "start_iso": {"type": "string"}, "end_iso": {"type": "string"}, "title": {"type": "string"}}, "required": ["organizer","participants","start_iso","end_iso"]}}},
    {"type": "function", "function": {"name": "confirm_meeting", "description": "Confirm meeting", "parameters": {"type": "object", "properties": {"meeting_id": {"type": "string"}}, "required": ["meeting_id"]}}},
]

async def calendar_freebusy(args):
    async with httpx.AsyncClient() as http:
        r = await http.post(f"{AVM_HTTP}/intents/calendar.freebusy", json=args, timeout=15)
        r.raise_for_status()
        return r.json()

async def propose_meeting(args):
    payload = {"organizer": args["organizer"], "participants": args["participants"], "slot": {"start_iso": args["start_iso"], "end_iso": args["end_iso"]}, "title": args.get("title", TITLE), "idem_key": str(uuid.uuid4())}
    async with httpx.AsyncClient() as http:
        r = await http.post(f"{AVM_HTTP}/intents/meeting.propose", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()

async def confirm_meeting(args):
    payload = {"meeting_id": args["meeting_id"], "idem_key": str(uuid.uuid4())}
    async with httpx.AsyncClient() as http:
        r = await http.post(f"{AVM_HTTP}/intents/meeting.confirm", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()

TOOL_IMPL = {"calendar_freebusy": calendar_freebusy, "propose_meeting": propose_meeting, "confirm_meeting": confirm_meeting}

SYSTEM_PROMPT = ("You are a careful scheduling agent. Find a 30-minute slot where ALL participants are free between the given window. Prefer earlier times. Then propose it via tools. If proposal is ok, confirm it. Respond with short status updates; always use the provided tools to act.")

async def subscribe_meeting(meeting_id: str):
    uri = f"{AVM_WS}/subscriptions/meeting:{meeting_id}"
    async with websockets.connect(uri) as ws:
        async for msg in ws:
            print("[projection]", json.loads(msg))

async def main():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"organizer": ORGANIZER, "participants": PARTICIPANTS, "window_start_iso": WINDOW_START, "window_end_iso": WINDOW_END, "duration_minutes": 30, "title": TITLE})}
    ]
    meeting_id = None
    while True:
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.1)
        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None) or []
        if tool_calls:
            for call in tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                print(f"[LLM->tool] {name} {args}")
                result = await TOOL_IMPL[name](args)
                messages.append({"role": "tool", "name": name, "content": json.dumps(result)})
                if name == "propose_meeting" and result.get("ok"):
                    meeting_id = result.get("correlation_id")
                    asyncio.create_task(subscribe_meeting(meeting_id))
            continue
        else:
            print("[LLM]", choice.message.content)
            if meeting_id:
                conf = await confirm_meeting({"meeting_id": meeting_id})
                print("[agent] confirmed:", conf)
            break

if __name__ == "__main__":
    asyncio.run(main())
```

Run the agent:

```bash
AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_KEY=... \
AZURE_OPENAI_API_VERSION=2024-07-01-preview AZURE_OPENAI_DEPLOYMENT=... \
python agent_llm.py
```

---

## 8) Optional Supervisor (supervisor.py)

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

## 9) End-to-End Test

1. Start the server:

   ```bash
   uvicorn server:app --reload
   ```
2. Start the agent:

   ```bash
   AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_KEY=... \
   AZURE_OPENAI_API_VERSION=2024-07-01-preview AZURE_OPENAI_DEPLOYMENT=... \
   python agent_llm.py
   ```
3. Observe the console:

   * LLM calls `calendar_freebusy`, reasons about options, calls `propose_meeting`.
   * Projection `meeting:M-*` is created and broadcast; agent subscribes and prints updates.
   * Agent confirms the meeting if appropriate.

---

## 10) Production Notes & Next Steps

* **Boundary safety**: LLM acts only through typed AVM tools.
* **Multi‑client**: Agents and UIs subscribe to the same projections.
* **Azure AI Foundry**: Swap in your deployment; add content filters and logging.
* Add **two‑phase participant approvals** and per‑client scopes/rate limits.
* Use **durable pub/sub** (Redis/NATS) behind the Projection Store for scale.
* Integrate Microsoft Graph/Google Calendar adapters and optimistic concurrency.

You now have a realistic AVM server and a **generative AI scheduling agent** using **Azure AI Foundry** tool calling in a separate process.
