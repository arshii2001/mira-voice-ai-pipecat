import re
from typing import List, Optional, Dict, Any
from agents.models import AgentSession

class StepController:
    """Manages parsing and tracking of curriculum teaching steps."""
    
    @staticmethod
    def parse_teaching_flow(flow_text: str) -> List[str]:
        """Extra individual steps from the 'Teaching Flow' block."""
        # Find the line "Teaching Flow (must follow in order):"
        # and parse the numbered lines below it
        try:
            # Extract content after the header
            parts = re.split(r"Teaching Flow \(must follow in order\):", flow_text, flags=re.IGNORECASE)
            if len(parts) < 2:
                # Handle alternative headers or missing header
                steps_raw = flow_text.strip().splitlines()
            else:
                steps_raw = parts[1].strip().splitlines()
            
            # Match lines starting with "1.", "2.", etc.
            steps = []
            for line in steps_raw:
                # Clean up whitespace and find numbered lines
                clean_line = line.strip()
                if re.match(r"^\d+\.", clean_line):
                    # Remove the number and keep the instructions
                    steps.append(re.sub(r"^\d+\.\s*", "", clean_line))
                elif not clean_line and steps:
                    # Allow blank lines but don't break the list
                    continue
            return steps
        except Exception:
            return [flow_text] # Fallback to full text if parsing fails

    @staticmethod
    def get_current_instruction(steps: List[str], current_step: int) -> str:
        """Get the specific instruction for the current step (1-indexed)."""
        idx = current_step - 1
        if 0 <= idx < len(steps):
            return steps[idx]
        return "Complete the topic with the student."

    @staticmethod
    def should_advance(step_instruction: str, user_input: str, tool_result: Any) -> bool:
        """Determine if we should advance to the next step based on tool results."""
        # This logic will be refined, but for now:
        # If the tool call was successful and matches the intended goal of the step
        # The agent pass will ultimately decide, but we can provide metadata here
        return True # Default to advancement for demo
