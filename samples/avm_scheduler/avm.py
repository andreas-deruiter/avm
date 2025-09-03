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