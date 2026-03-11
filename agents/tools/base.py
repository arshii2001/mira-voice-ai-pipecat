from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel
from agents.models import ToolResult

class BaseMathTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tool for the agent."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description for the agent's tool selector."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema of the tool's input parameters."""
        pass

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """Execute the tool's logic."""
        pass
