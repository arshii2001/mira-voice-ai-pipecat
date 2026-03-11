from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import time

class ProblemStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SOLVED = "solved"
    FAILED = "failed"

class AgentSession(BaseModel):
    id: str
    topic_id: str
    current_step: int = 1
    student_name: Optional[str] = None
    attempts_at_step: int = 0
    problem_state: Dict[str, Any] = Field(default_factory=dict)
    status: ProblemStatus = ProblemStatus.IN_PROGRESS
    last_updated: float = Field(default_factory=time.time)

class ToolResult(BaseModel):
    success: bool
    value: Any
    error: Optional[str] = None
    steps: List[str] = Field(default_factory=list)

class MathTrace(BaseModel):
    request_id: str
    user_input: str
    extraction: Dict[str, Any] = Field(default_factory=dict)
    tool: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)
    tutor_prompt_mode: str = "hook"
    latency_ms: Dict[str, float] = Field(default_factory=dict)
