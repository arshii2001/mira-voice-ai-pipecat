import json
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx

from agents.models import AgentSession, ProblemStatus, MathTrace
from agents.session_store import BaseSessionStore, get_default_session_store
from agents.tools import global_tool_router
from agents.step_controller import StepController
from maths_manager import get_maths_manager
from provider_config import LLM

logger = logging.getLogger(__name__)

class MathAgent:
    def __init__(self, session_store: Optional[BaseSessionStore] = None):
        self.session_store = session_store or get_default_session_store()
        self.tool_router = global_tool_router
        self.maths_manager = get_maths_manager()

    async def _extract_math(self, user_input: str, context_topic: str) -> Dict[str, Any]:
        """Pass 1: Extract math expression from natural language."""
        prompt = (
            "You are a mathematical entity extractor. Your goal is to extract the math expression "
            "from the student's message. Focus ONLY on the numbers and operators.\n\n"
            f"Topic Context: {context_topic}\n"
            f"Student Message: {user_input}\n\n"
            "Respond in JSON format: {\"expression\": \"...\", \"type\": \"arithmetic|number|null\"}"
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{LLM.base_url}/chat/completions",
                    json={
                        "model": LLM.model,
                        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_input}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.0
                    },
                    headers={"Authorization": f"Bearer {LLM.api_key}"},
                )
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {"expression": None, "type": "null"}

    async def handle_chat(self, session_id: str, user_input: str, topic_id: str, student_name: Optional[str] = None, history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """Main orchestrator loop for handling a student chat message."""
        t0 = time.time()
        
        try:
            # 1. Load Session
            session = await self.session_store.get_session(session_id)
            if not session:
                session = AgentSession(id=session_id, topic_id=topic_id, student_name=student_name)
                await self.session_store.save_session(session)

            # 2. Get Curriculum Context
            topic_info = self.maths_manager.get_topic(topic_id)
            if not topic_info:
                yield "I couldn't find that math topic index. Let's try another one!"
                return

            teaching_flow = topic_info.get("reasoning", "")
            steps = StepController.parse_teaching_flow(teaching_flow)
            current_step_instruction = StepController.get_current_instruction(steps, session.current_step)

            # 3. Extraction (Pass 1)
            extraction_t0 = time.time()
            # Optimization: Check if input is just a number first
            if user_input.strip().isdigit():
                extraction = {"expression": user_input.strip(), "type": "number"}
            else:
                extraction = await self._extract_math(user_input, topic_info.get("improved_topic_name", ""))
            extraction_ms = (time.time() - extraction_t0) * 1000

            # 4. Tool Execution
            tool_t0 = time.time()
            tool_result = None
            if extraction.get("expression"):
                tool_result = await self.tool_router.call_tool("math_verifier", expression=extraction["expression"])
            tool_ms = (time.time() - tool_t0) * 1000

            # 5. Validation & State Management
            if tool_result and tool_result.success:
                # Update current step if the student solved the current instruction
                # (Placeholder logic for advancement)
                logger.info(f"Step {session.current_step} verified with value {tool_result.value}")

            # 6. Tutor Response (Pass 2) - Streaming
            tutor_t0 = time.time()
            
            # Load the base maths-agent prompt
            from bot import load_system_prompt, PROMPT_VERSION
            base_prompt = load_system_prompt(version=PROMPT_VERSION, mode="maths-agent")
            
            # Inject Student Context
            student_context = f"\n\n--- STUDENT CONTEXT ---\nName: {student_name or 'Friend'}\nTopic: {topic_info.get('improved_topic_name', topic_info.get('original_topic'))}\n"
            
            # Inject Curriculum Context
            curriculum_context = self.maths_manager.get_context_for_topic(topic_id) or ""
            
            # Inject Agent Metadata (Invisible to user)
            metadata_block = (
                f"\n\n--- AGENT METADATA ---\n"
                f"Verified Answer: {tool_result.value if (tool_result and tool_result.success) else 'Unknown'}\n"
                f"Curriculum Step Instruction: {current_step_instruction}\n"
                f"Note: If history is provided, do NOT repeat the introduction. Respond contextually to the latest message.\n"
            )
            
            system_prompt = base_prompt + student_context + curriculum_context + metadata_block

            # Prepare messages with history
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                # Truncate history to avoid token bloat
                MAX_HISTORY = 10
                for m in history[-MAX_HISTORY:]:
                    # Don't include redundant system prompts from previous turns if they were saved in history
                    if m["role"] != "system":
                        messages.append({"role": m["role"], "content": m["content"]})
            else:
                messages.append({"role": "user", "content": user_input})

            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{LLM.base_url}/chat/completions",
                    json={
                        "model": LLM.model,
                        "messages": messages,
                        "stream": True,
                        "temperature": 0.7
                    },
                    headers={"Authorization": f"Bearer {LLM.api_key}"},
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0]["delta"]
                                if "content" in delta:
                                    yield delta["content"]
                            except:
                                continue

            # 7. Update Session
            await self.session_store.save_session(session)
            logger.info(f"MathAgent Trace | Ext: {extraction_ms:.1f}ms | Tool: {tool_ms:.1f}ms | Total: {(time.time()-t0)*1000:.1f}ms")

        except Exception as e:
            logger.error(f"Error in MathAgent.handle_chat: {e}", exc_info=True)
            yield f"I ran into an issue while processing your request: {str(e)}. Please try again!"
