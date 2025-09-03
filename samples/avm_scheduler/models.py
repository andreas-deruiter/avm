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