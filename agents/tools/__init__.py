from typing import Dict, Any, Optional
from agents.tools.base import BaseMathTool, ToolResult
from agents.tools.math_verifier import MathVerifierTool

class ToolRouter:
    def __init__(self):
        self._tools: Dict[str, BaseMathTool] = {
            "math_verifier": MathVerifierTool()
        }

    def get_tool(self, name: str) -> Optional[BaseMathTool]:
        return self._tools.get(name)

    async def call_tool(self, name: str, **kwargs) -> ToolResult:
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(success=False, value=None, error=f"Tool not found: {name}")
        
        try:
            return await tool.run(**kwargs)
        except Exception as e:
            return ToolResult(success=False, value=None, error=f"Tool execution error: {str(e)}")

# Global router instance
global_tool_router = ToolRouter()
