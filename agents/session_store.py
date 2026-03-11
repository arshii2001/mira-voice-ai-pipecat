import time
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict
from agents.models import AgentSession

class BaseSessionStore(ABC):
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[AgentSession]:
        pass

    @abstractmethod
    async def save_session(self, session: AgentSession) -> None:
        pass

    @abstractmethod
    async def update_session(self, session_id: str, updates: dict) -> None:
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        pass

class InMemorySessionStore(BaseSessionStore):
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self._sessions: Dict[str, AgentSession] = {}
        self._max_size = max_size
        self._ttl = ttl  # TTL in seconds

    async def get_session(self, session_id: str) -> Optional[AgentSession]:
        session = self._sessions.get(session_id)
        if session:
            # Check TTL
            if time.time() - session.last_updated > self._ttl:
                await self.delete_session(session_id)
                return None
            return session
        return None

    async def save_session(self, session: AgentSession) -> None:
        session.last_updated = time.time()
        self._sessions[session.id] = session
        
        # Simple LRU-ish eviction if we hit max_size
        if len(self._sessions) > self._max_size:
            # Sort by last updated and delete oldest
            oldest_id = min(self._sessions.keys(), key=lambda k: self._sessions[k].last_updated)
            del self._sessions[oldest_id]

    async def update_session(self, session_id: str, updates: dict) -> None:
        session = await self.get_session(session_id)
        if session:
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.last_updated = time.time()
            self._sessions[session_id] = session

    async def delete_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

# Singleton instance for in-memory store
_global_store = InMemorySessionStore()

def get_default_session_store() -> BaseSessionStore:
    return _global_store
