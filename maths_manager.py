"""
Maths Curriculum Manager
Handles loading and serving content from the flattened maths curriculum JSON (e.g. 3_counting.json).
Provides O(1) access to grades and topics.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Default path relative to this file or from env
CURRICULUM_DIR = os.environ.get("CURRICULUM_DIR", "/content/curriculum")
DEFAULT_MATHS_FILE = os.path.join(CURRICULUM_DIR, "3_counting_final.json")

class MathsManager:
    def __init__(self, json_path: Optional[str] = None):
        self._path = json_path or os.environ.get("MATHS_CURRICULUM_FILE", DEFAULT_MATHS_FILE)
        
        # Data storage
        # grades: sorted list of available grades [3, 4, 5]
        self._grades: List[int] = []
        
        # topics_by_grade: { 3: [topic_dict, ...], 4: [...] }
        self._topics_by_grade: Dict[int, List[Dict[str, Any]]] = {}
        
        # topics_by_id: { "1": topic_dict, "2": topic_dict } (using index as ID)
        self._topics_by_id: Dict[str, Dict[str, Any]] = {}

        self._load_data()

    def _load_data(self):
        """Load the JSON file and build indexes."""
        path = Path(self._path)
        if not path.exists():
            logger.warning(f"[MATHS] File not found: {path} — maths features disabled")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error(f"[MATHS] Invalid JSON structure in {path}. Expected a list.")
                return

            grades_set = set()
            
            for item in data:
                # Validate essential fields
                idx = item.get("index")
                grade = item.get("grade")
                
                if idx is None or grade is None:
                    continue
                
                # Normalize IDs to string
                topic_id = str(idx)
                # Normalize grade to int (handle "3" vs 3)
                try:
                    grade_int = int(grade)
                except ValueError:
                    continue

                grades_set.add(grade_int)
                
                # Store by ID
                self._topics_by_id[topic_id] = item
                
                # Store by Grade
                if grade_int not in self._topics_by_grade:
                    self._topics_by_grade[grade_int] = []
                self._topics_by_grade[grade_int].append(item)

            self._grades = sorted(list(grades_set))
            logger.info(f"[MATHS] Loaded {len(self._topics_by_id)} topics across grades {self._grades}")

        except Exception as e:
            logger.error(f"[MATHS] Failed to load {path}: {e}")

    @property
    def available(self) -> bool:
        return len(self._topics_by_id) > 0

    def get_grades(self) -> List[int]:
        """Return list of available grades."""
        return self._grades

    def get_topics_for_grade(self, grade: int) -> List[Dict[str, Any]]:
        """Return list of topics (summary info) for a specific grade."""
        topics = self._topics_by_grade.get(grade, [])
        # Return a simplified version for the UI if needed, or the full object
        # For now, return full object but maybe strictly typed in future
        return topics

    def get_topic(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """Return full topic details by ID."""
        return self._topics_by_id.get(str(topic_id))

    def get_context_for_topic(self, topic_id: str) -> Optional[str]:
        """
        Build a context string for the LLM system prompt.
        Format includes: Topic Name, Summary, Objectives, Sample Q&A.
        """
        topic = self.get_topic(topic_id)
        if not topic:
            return None

        lines = []
        lines.append(f"Subject: Mathematics (Grade {topic.get('grade')})")
        lines.append(f"Topic: {topic.get('improved_topic_name', topic.get('original_topic'))}")
        
        summary = topic.get('topic_summary')
        if summary:
            lines.append(f"Summary: {summary}")
            
        objective = topic.get('learning_objective_summary')
        if objective:
            lines.append(f"Learning Objectives: {objective}")
            
        reasoning = topic.get('reasoning')
        if reasoning:
            lines.append(f"\nTeaching Strategy:\n{reasoning}")
        
        return "\n".join(lines)

# Singleton
_instance: Optional[MathsManager] = None

def get_maths_manager() -> MathsManager:
    global _instance
    if _instance is None:
        _instance = MathsManager()
    return _instance
