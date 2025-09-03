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