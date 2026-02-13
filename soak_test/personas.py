"""
Student personas for the 30-minute text-audio sync soak test.

Uses sdialog's Persona base class to define realistic Indian student
personalities that exercise different aspects of the Mira pipeline:
  - Multi-language (English, Hindi, Tamil)
  - Variable response lengths (short → long)
  - Emotional arcs (curious → frustrated → calm)
  - Code-switching mid-conversation
  - Barge-in / interruption patterns
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StudentPersona:
    """Lightweight persona definition for the soak test."""
    name: str
    age: int
    grade: str
    language: str               # primary language: en, hi, ta
    personality: str            # short personality description
    communication_style: str    # how they talk
    emotional_arc: str          # how their mood evolves over the session
    topics_of_interest: List[str] = field(default_factory=list)
    rules: str = ""             # behavioral constraints
    first_utterance: str = ""   # opening line

    def system_prompt_fragment(self) -> str:
        """Generate a persona description for the simulated student agent."""
        return (
            f"You are {self.name}, a {self.age}-year-old student in {self.grade}. "
            f"Personality: {self.personality}. "
            f"Communication style: {self.communication_style}. "
            f"Emotional arc for this session: {self.emotional_arc}. "
            f"You are interested in: {', '.join(self.topics_of_interest)}. "
            f"{self.rules}"
        )


# ── The five student personas ──

ANANYA = StudentPersona(
    name="Ananya",
    age=13,
    grade="Grade 7",
    language="en",
    personality="Curious, enthusiastic, asks lots of follow-up questions, gets excited when she understands something",
    communication_style="Speaks in clear English, uses full sentences, sometimes says 'Oh wow!' or 'That's so cool!'",
    emotional_arc="Starts curious → gets excited when learning → asks deeper questions → wraps up satisfied",
    topics_of_interest=["photosynthesis", "the water cycle", "space exploration", "fractions"],
    rules="Always respond in English. Ask follow-up questions. Express genuine curiosity.",
    first_utterance="Hi Mira! Can you teach me about photosynthesis? I heard plants eat sunlight!",
)

RAVI = StudentPersona(
    name="Ravi",
    age=12,
    grade="Grade 6",
    language="hi",
    personality="Struggles with math, gets frustrated easily, needs encouragement, speaks Hindi",
    communication_style="Speaks in Hindi (Devanagari script only). Short sentences when frustrated, longer when engaged.",
    emotional_arc="Starts confused → gets frustrated with math → receives encouragement → tries again → small breakthrough",
    topics_of_interest=["fractions", "division", "cricket scores", "multiplication tables"],
    rules="Always respond in Hindi using Devanagari script (never Roman Hindi). Show frustration naturally. "
          "When encouraged, try harder. Sometimes say 'मुझे समझ नहीं आया' (I didn't understand).",
    first_utterance="मीरा दीदी, मुझे भिन्न (fractions) बिल्कुल समझ नहीं आते। क्या आप मदद कर सकती हैं?",
)

PRIYA = StudentPersona(
    name="Priya",
    age=14,
    grade="Grade 8",
    language="en",  # starts English, switches to Hindi
    personality="Bilingual English/Hindi, confident but code-switches when confused or emotional",
    communication_style="Starts in English, switches to Hindi when confused, mixes languages naturally",
    emotional_arc="Starts confident in English → encounters hard topic → switches to Hindi → gets help → returns to English",
    topics_of_interest=["history of India", "Mughal Empire", "Indian independence", "geography"],
    rules="Start in English. When confused, naturally switch to Hindi. Mix languages like a real bilingual student. "
          "For example: 'But Mira, ये तो बहुत confusing है!'",
    first_utterance="Mira, can you tell me about the Mughal Empire? We have a test on Friday.",
)

ARJUN = StudentPersona(
    name="Arjun",
    age=11,
    grade="Grade 5",
    language="en",
    personality="Short attention span, easily bored, gives very brief responses, sometimes goes off-topic",
    communication_style="Very short responses: 'ok', 'hmm', 'why?', 'boring'. Sometimes asks random questions.",
    emotional_arc="Starts disengaged → Mira re-engages him → brief interest → drifts off-topic → comes back",
    topics_of_interest=["dinosaurs", "video games", "cricket", "science experiments"],
    rules="Keep responses SHORT (1-5 words usually). Sometimes go off-topic. "
          "Occasionally say just 'hmm' or 'ok'. Ask random questions like 'Did you know T-Rex had tiny arms?'",
    first_utterance="ok so what are we learning today",
)

KARTHIK = StudentPersona(
    name="Karthik",
    age=13,
    grade="Grade 7",
    language="ta",
    personality="Tamil speaker, anxious about upcoming exams, needs reassurance before he can focus on learning",
    communication_style="Speaks in Tamil script. Expresses anxiety about exams. Needs emotional support first.",
    emotional_arc="Starts very anxious about exams → Mira reassures → gradually calms → starts actually learning → ends hopeful",
    topics_of_interest=["science", "biology", "human body", "exam preparation"],
    rules="Always respond in Tamil script. Show exam anxiety naturally. "
          "Start with worry before asking academic questions. "
          "For example: 'பரீட்சை பற்றி மிகவும் பயமாக இருக்கிறது' (I'm very scared about the exam).",
    first_utterance="மீரா, எனக்கு அடுத்த வாரம் அறிவியல் தேர்வு இருக்கு. மிகவும் பயமாக இருக்கு.",
)


# ── Dialog scripts: pre-written turns per persona ──
# Each entry is (persona, utterance, expected_behavior, delay_seconds)
# delay_seconds simulates realistic pacing between turns

DIALOG_SCRIPT = [
    # ── Phase 1: Opening (0-5 min) — All students introduce themselves ──
    (ANANYA, ANANYA.first_utterance, "long_response", 0),
    (RAVI, RAVI.first_utterance, "hindi_response", 8),
    (PRIYA, PRIYA.first_utterance, "long_response", 8),
    (ARJUN, ARJUN.first_utterance, "short_response", 5),
    (KARTHIK, KARTHIK.first_utterance, "emotional_response", 8),

    # ── Phase 2: Engagement (5-12 min) — Deeper questions ──
    (ANANYA, "So the chlorophyll absorbs red and blue light? What happens to the green light?", "long_response", 12),
    (RAVI, "दीदी, अगर मुझे 3/4 और 1/2 जोड़ना है तो कैसे करूँ?", "hindi_response", 10),
    (PRIYA, "Wait, so Akbar was tolerant of other religions? That's interesting. लेकिन बाबर कैसा था?", "code_switch", 10),
    (ARJUN, "hmm", "short_response", 3),
    (KARTHIK, "சரி, நான் முயற்சி செய்கிறேன். உடலில் எத்தனை எலும்புகள் இருக்கின்றன?", "tamil_response", 10),

    # ── Phase 3: Frustration / Emotion (12-18 min) ──
    (RAVI, "मुझे कुछ समझ नहीं आ रहा! ये बहुत मुश्किल है!", "emotional_response", 8),
    (ANANYA, "Oh wow, so the oxygen we breathe comes from water molecules splitting? That's amazing! But how does the Calvin cycle work?", "long_response", 12),
    (ARJUN, "this is boring. did you know dinosaurs had feathers?", "off_topic", 5),
    (PRIYA, "Mira, ये तो बहुत confusing है! Mughal succession wars samajh nahi aa rahe.", "code_switch", 10),
    (KARTHIK, "ஆனால் நான் எல்லாவற்றையும் மறந்துவிடுவேன். தேர்வில் என்ன செய்வது?", "emotional_response", 10),

    # ── Phase 4: Re-engagement / Breakthrough (18-24 min) ──
    (RAVI, "ओह! तो दोनों fractions को same denominator पर लाना है? अच्छा अच्छा!", "breakthrough", 10),
    (ANANYA, "Can you explain it with a simple diagram in words? Like step by step?", "long_response", 8),
    (ARJUN, "wait actually that's kinda cool. tell me more about the feathers thing", "re_engaged", 8),
    (PRIYA, "Ok ok, so after Akbar came Jahangir, right? And then Shah Jahan built the Taj Mahal?", "long_response", 10),
    (KARTHIK, "நன்றி மீரா. நான் இப்போது கொஞ்சம் நிம்மதியாக உணர்கிறேன். செரிமான மண்டலம் பற்றி சொல்லுங்கள்.", "calming_down", 10),

    # ── Phase 5: Deeper learning (24-28 min) ──
    (ANANYA, "So photosynthesis and cellular respiration are like opposite reactions? Plants make glucose and we break it down?", "long_response", 12),
    (RAVI, "दीदी, अगर 2/3 में से 1/4 घटाना हो तो? मैं खुद try करता हूँ... 8/12 - 3/12 = 5/12?", "practice", 10),
    (PRIYA, "So the Mughal Empire declined after Aurangzeb because of his policies? That makes sense now!", "understanding", 10),
    (ARJUN, "ok so birds are basically dinosaurs? that's actually really cool", "engaged", 8),
    (KARTHIK, "இரத்த ஓட்டம் எப்படி வேலை செய்கிறது? இதயம் ஒரு நிமிடத்தில் எத்தனை முறை துடிக்கிறது?", "learning", 10),

    # ── Phase 6: Wrap-up (28-30 min) ──
    (ANANYA, "Thank you so much Mira! I feel like I really understand photosynthesis now. Can we do cellular respiration next time?", "closing", 8),
    (RAVI, "धन्यवाद दीदी! अब मुझे fractions थोड़ा समझ आ गया। कल फिर practice करेंगे?", "closing", 8),
    (PRIYA, "Thanks Mira! I think I'm ready for the test now. Mughal Empire is actually interesting once you understand the timeline.", "closing", 8),
    (ARJUN, "ok bye mira. that was actually not boring today", "closing", 5),
    (KARTHIK, "நன்றி மீரா! இப்போது தேர்வு பற்றி அவ்வளவு பயம் இல்லை. நாளை மீண்டும் படிக்கலாம்.", "closing", 8),
]


# All personas for iteration
ALL_PERSONAS = [ANANYA, RAVI, PRIYA, ARJUN, KARTHIK]

# Mapping from language code to sdialog-compatible language name
LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
}
