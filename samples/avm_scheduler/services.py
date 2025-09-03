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