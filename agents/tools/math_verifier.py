import sympy
from typing import Any, Dict, List
from agents.tools.base import BaseMathTool, ToolResult

class MathVerifierTool(BaseMathTool):
    @property
    def name(self) -> str:
        return "math_verifier"

    @property
    def description(self) -> str:
        return (
            "Verify a mathematical expression (arithmetic, fractions, simple algebra). "
            "Input should be a string like '15 + 4' or '1/2 + 1/4'. "
            "Returns the evaluated result."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate."
                }
            },
            "required": ["expression"]
        }

    def _sanitize_expression(self, expression: str) -> str:
        """Whitelist only mathematical characters and simplify the expression string."""
        # Simple whitelist: numbers, operators, parens, decimal points, and 'x' for algebra
        allowed_chars = set("0123456789+-*/() .x")
        sanitized = "".join(c for c in expression if c in allowed_chars).strip()
        
        # Max length to prevent ReDoS
        if len(sanitized) > 100:
            raise ValueError("Expression too long")
        return sanitized

    async def run(self, **kwargs) -> ToolResult:
        expression = kwargs.get("expression", "")
        if not expression:
            return ToolResult(success=False, value=None, error="No expression provided")

        try:
            sanitized = self._sanitize_expression(expression)
            
            # Using a restricted namespace for SymPy
            # Only allow essential math objects
            limit_namespace = {
                'Integer': sympy.Integer,
                'Rational': sympy.Rational,
                'Add': sympy.Add,
                'Mul': sympy.Mul,
                'Sub': sympy.Add, # sub is add with -1
                'Pow': sympy.Pow,
                'Symbol': sympy.Symbol,
                'x': sympy.Symbol('x')
            }
            
            # Sympify the expression
            # Use parse_expr with custom transformations if we want even more control
            # For now, simple sympify is okay since we sanitized the input string
            result = sympy.sympify(sanitized, locals=limit_namespace, evaluate=True)
            
            # Convert result to a clean string or float for the agent
            if isinstance(result, sympy.Rational):
                clean_value = str(result)
            elif result.is_integer:
                clean_value = int(result)
            else:
                clean_value = float(result)

            return ToolResult(
                success=True, 
                value=clean_value, 
                error=None,
                steps=[f"Expression: {sanitized}", f"Result: {clean_value}"]
            )
            
        except Exception as e:
            return ToolResult(success=False, value=None, error=f"Math error: {str(e)}")
