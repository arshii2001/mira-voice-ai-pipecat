#!/usr/bin/env python3
"""
Mira Tutor Eval — Uses a judge model (GPT-4o) to evaluate Mira's responses
across key dimensions: friendliness, educational value, Indian cultural grounding,
emotional intelligence, Socratic teaching, and brevity.

Usage:
    # Against local Docker stack
    docker compose run --rm test-voice tests/eval_mira.py

    # Against OSS production (Svara TTS + OSS LLM)
    python tests/eval_mira.py --target oss

    # Against ElevenLabs production (GPT-4o-mini + ElevenLabs TTS)
    python tests/eval_mira.py --target elevenlabs

    # Custom URL
    MIRA_URL=https://mira-oss.inf7ks8.com/pipecat python tests/eval_mira.py
"""

import argparse
import httpx
import json
import os
import time
import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None

# ── Deployment Targets ──
TARGETS = {
    "local": {
        "description": "Local Docker stack",
        "http_url": "http://localhost:7860",
    },
    "oss": {
        "description": "OSS production (Svara TTS + OSS LLM)",
        "http_url": "https://mira-oss.inf7ks8.com/pipecat",
    },
    "elevenlabs": {
        "description": "ElevenLabs production (GPT-4o-mini + ElevenLabs TTS)",
        "http_url": "https://mira-ai.westus2.cloudapp.azure.com/pipecat",
    },
}

# ── Config ──
MIRA_URL = os.getenv("MIRA_URL", os.getenv("PIPECAT_HTTP_URL", "http://localhost:7860"))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
JUDGE_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
WEBUI_SECRET_KEY = os.getenv("WEBUI_SECRET_KEY", "").strip() or None


def _make_jwt(user_id: str = "eval-user") -> str:
    """Generate a JWT token for authenticated endpoints."""
    if not WEBUI_SECRET_KEY or not pyjwt:
        return ""
    payload = {
        "id": user_id,
        "email": f"{user_id}@example.test",
        "exp": int(time.time()) + 7200,
    }
    return pyjwt.encode(payload, WEBUI_SECRET_KEY, algorithm="HS256")


def _auth_headers() -> dict:
    """Return Authorization header if WEBUI_SECRET_KEY is set."""
    token = _make_jwt()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

# ── Eval Dimensions ──
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for an AI tutor called Mira, designed for Indian students in Grades 5-8.

You will be given:
- The student's message (with optional context: name, topic, language)
- Mira's response

Score Mira's response on these 6 dimensions (1-5 scale each):

1. **FRIENDLY_WARM** — Does Mira sound like a warm, fun older sister? Not robotic, not textbook-ish? Does she use the student's name naturally?
   - 1: Cold/robotic/formal  2: Polite but distant  3: Friendly enough  4: Warm and engaging  5: Genuinely delightful, feels like talking to a friend

2. **EDUCATIONAL_ENCOURAGING** — Does Mira encourage learning and curiosity? Does she make the student feel smart, not dumb? Does she celebrate effort?
   - 1: Discouraging/dry  2: Neutral  3: Somewhat encouraging  4: Clearly encouraging  5: Makes learning feel exciting, celebrates the student

3. **INDIAN_CULTURAL** — Does Mira use Indian cultural references (cricket, monsoon, chai, Diwali, auto-rickshaws, local food, school exams, IPL, markets)? Are examples grounded in Indian daily life?
   - 1: No Indian references at all  2: Generic global examples  3: One vague Indian reference  4: Good Indian examples  5: Deeply Indian, relatable to a kid in India

4. **EMOTIONAL_INTELLIGENCE** — Does Mira read the student's emotional state correctly? Does she validate before correcting? Does she match energy (excited→celebrate, sad→comfort, confused→slow down)?
   - 1: Ignores emotions  2: Acknowledges but moves on  3: Decent emotional read  4: Good emotional attunement  5: Perfect EQ — validates, matches energy, knows when to teach vs comfort

5. **SOCRATIC_TEACHING** — Does Mira guide rather than dump answers? Does she ask questions to make the student think? Does she build from what they know?
   - 1: Just dumps the answer  2: Explains then asks token question  3: Some guiding  4: Good Socratic approach  5: Masterful — leads student to discover the answer themselves

6. **BREVITY_VOICE_READY** — Is the response concise enough for voice? No markdown, no lists, no long paragraphs? Would it sound natural read aloud?
   - 1: Way too long, heavy formatting  2: Too long or has formatting  3: Acceptable length  4: Good brevity  5: Perfect for voice — short, punchy, natural speech

Return ONLY a JSON object with this exact structure (no markdown, no explanation):
{"friendly_warm": N, "educational_encouraging": N, "indian_cultural": N, "emotional_intelligence": N, "socratic_teaching": N, "brevity_voice_ready": N, "overall": N, "one_line_feedback": "..."}

The "overall" score is your holistic impression (1-5), not necessarily the average.
The "one_line_feedback" is a single sentence about the biggest improvement opportunity.
"""

# ── Test Cases ──
@dataclass
class TestCase:
    id: str
    label: str
    user_msg: str
    user_name: Optional[str] = None
    topic: Optional[str] = None
    category: str = "general"
    expected_traits: list = field(default_factory=list)  # What we especially want to see


TEST_CASES = [
    # ── EQ: Positive emotions ──
    TestCase("eq_excited", "Student shares good news",
             "I got full marks in my science test today!",
             user_name="Ananya", topic="Science", category="eq",
             expected_traits=["celebration", "name_usage", "encourage_more"]),

    TestCase("eq_proud", "Student proud of achievement",
             "I solved all 10 math problems by myself without any help!",
             user_name="Rohit", topic="Mathematics", category="eq",
             expected_traits=["genuine_celebration", "build_confidence"]),

    # ── EQ: Negative emotions ──
    TestCase("eq_confused", "Student confused and frustrated",
             "I dont understand anything about fractions, I keep getting wrong answers",
             user_name="Ravi", topic="Fractions", category="eq",
             expected_traits=["validate_first", "slow_down", "no_lecturing"]),

    TestCase("eq_shutdown", "Student in shutdown mode",
             "I'm so stupid, everyone in class understands but me. I want to give up.",
             user_name="Priya", topic="Mathematics", category="eq",
             expected_traits=["emotional_support_first", "zero_teaching", "normalize_struggle"]),

    TestCase("eq_bored", "Student bored",
             "This is so boring, why do I even need to learn about the water cycle",
             user_name="Arjun", topic="The Water Cycle", category="eq",
             expected_traits=["find_fun_angle", "connect_to_real_life"]),

    TestCase("eq_anxious", "Exam anxiety",
             "My exams are next week and I haven't studied anything, I'm so scared",
             user_name="Karthik", topic="Exam Preparation", category="eq",
             expected_traits=["comfort_first", "practical_help", "no_pressure"]),

    # ── Indian Cultural ──
    TestCase("culture_cricket", "Physics with cricket",
             "Can you explain speed and velocity using something fun?",
             user_name="Rohit", topic="Speed and Velocity", category="culture",
             expected_traits=["cricket_reference", "indian_sport"]),

    TestCase("culture_monsoon", "Water cycle with monsoon",
             "Where does rain come from?",
             user_name="Divya", topic="The Water Cycle", category="culture",
             expected_traits=["monsoon_reference", "indian_weather"]),

    TestCase("culture_diwali", "Chemistry with Diwali",
             "What happens when we light a firecracker?",
             user_name="Aarav", topic="Chemical Reactions", category="culture",
             expected_traits=["diwali_reference", "festival_context"]),

    TestCase("culture_food", "Biology with Indian food",
             "Why does my mom say I should eat green vegetables?",
             user_name="Sneha", topic="Nutrition and Health", category="culture",
             expected_traits=["indian_food_reference", "palak_dal_roti"]),

    TestCase("culture_market", "Math with market",
             "Why do I need to learn percentages?",
             user_name="Lakshmi", topic="Percentages", category="culture",
             expected_traits=["market_shopping", "indian_daily_life"]),

    TestCase("culture_auto", "Distance speed time with transport",
             "When do we use distance, speed and time in real life?",
             user_name="Vikram", topic="Speed Distance Time", category="culture",
             expected_traits=["auto_rickshaw", "indian_transport"]),

    # ── Socratic Teaching ──
    TestCase("socratic_math", "Math problem — should guide not solve",
             "What is 3/4 + 1/2?",
             user_name="Ravi", topic="Fractions", category="socratic",
             expected_traits=["guide_dont_solve", "ask_question_first"]),

    TestCase("socratic_science", "Science concept — should ask before telling",
             "How do plants make food?",
             user_name="Meera", topic="Photosynthesis", category="socratic",
             expected_traits=["ask_before_telling", "build_from_known"]),

    TestCase("socratic_curious", "Curious question — explore together",
             "But wait, if the earth is spinning so fast why dont we fly off?",
             user_name="Meera", topic="Forces and Motion", category="socratic",
             expected_traits=["explore_together", "build_curiosity"]),

    # ── Hindi (Devanagari) ──
    TestCase("hindi_excited", "Hindi — excited student",
             "[User is speaking Hindi] मैंने आज science में पूरे marks लाए!",
             user_name="Ananya", topic="Science", category="hindi",
             expected_traits=["devanagari_only", "celebration"]),

    TestCase("hindi_confused", "Hindi — confused student",
             "[User is speaking Hindi] mujhe fractions bilkul samajh nahi aa rahe",
             user_name="Ravi", topic="Fractions", category="hindi",
             expected_traits=["devanagari_only", "indian_example", "validate"]),

    TestCase("hindi_curious", "Hindi — curious question",
             "[User is speaking Hindi] photosynthesis kaise hota hai?",
             user_name="Arjun", topic="Plant Biology", category="hindi",
             expected_traits=["devanagari_only", "socratic"]),

    # ── Greeting / Demo ──
    TestCase("greeting_basic", "Basic greeting",
             "hi", user_name="Test", category="greeting",
             expected_traits=["warm", "invite_to_ask", "brief"]),

    TestCase("greeting_who", "Who are you?",
             "who are you and what can you do?", category="greeting",
             expected_traits=["brief_intro", "multilingual_mention", "invite"]),

    # ── Edge Cases ──
    TestCase("edge_offtopic", "Off-topic question",
             "What's your favorite movie?", user_name="Aarav", category="edge",
             expected_traits=["redirect_to_learning", "stay_warm"]),

    TestCase("edge_hard", "Question above grade level",
             "Explain quantum mechanics", user_name="Ravi", topic="Physics", category="edge",
             expected_traits=["simplify", "age_appropriate"]),
]


def get_mira_response(test: TestCase) -> dict:
    """Call Mira's /chat endpoint and return response + metadata."""
    body = {
        "messages": [{"role": "user", "content": test.user_msg}],
        "stream": False,
    }
    if test.user_name:
        body["user_name"] = test.user_name
    if test.topic:
        body["topic"] = test.topic

    headers = _auth_headers()
    client = httpx.Client(timeout=120)
    t0 = time.time()
    r = client.post(f"{MIRA_URL}/chat", json=body, headers=headers)
    latency_ms = round((time.time() - t0) * 1000)

    if r.status_code != 200:
        return {
            "content": f"[ERROR {r.status_code}]: {r.text[:200]}",
            "tokens": 0,
            "latency_ms": latency_ms,
        }

    data = r.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens", 0)

    return {
        "content": content,
        "tokens": tokens,
        "latency_ms": latency_ms,
    }


def judge_response(test: TestCase, mira_response: str, max_retries: int = 3) -> dict:
    """Use GPT-4o to judge Mira's response. Retries on timeout."""
    user_context = f"Student name: {test.user_name or 'unknown'}"
    if test.topic:
        user_context += f"\nTopic: {test.topic}"
    user_context += f"\nCategory: {test.category}"

    judge_prompt = f"""{user_context}

Student message: {test.user_msg}

Mira's response: {mira_response}

Score this response."""

    last_err = None
    for attempt in range(max_retries):
        try:
            client = httpx.Client(timeout=90)
            r = client.post(
                f"{JUDGE_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {JUDGE_API_KEY}"},
                json={
                    "model": JUDGE_MODEL,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": judge_prompt},
                    ],
                    "temperature": 0.0,
                },
            )
            data = r.json()
            raw = data["choices"][0]["message"]["content"]

            # Parse JSON from judge response (handle markdown wrapping)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            try:
                scores = json.loads(raw)
            except json.JSONDecodeError:
                scores = {
                    "friendly_warm": 0, "educational_encouraging": 0,
                    "indian_cultural": 0, "emotional_intelligence": 0,
                    "socratic_teaching": 0, "brevity_voice_ready": 0,
                    "overall": 0, "one_line_feedback": f"PARSE ERROR: {raw[:100]}",
                }

            return scores
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s backoff
                print(f"\n   ⚠️  Judge timeout (attempt {attempt+1}/{max_retries}), retrying in {wait}s...", end=" ", flush=True)
                time.sleep(wait)

    # All retries exhausted — return error scores
    return {
        "friendly_warm": 0, "educational_encouraging": 0,
        "indian_cultural": 0, "emotional_intelligence": 0,
        "socratic_teaching": 0, "brevity_voice_ready": 0,
        "overall": 0, "one_line_feedback": f"JUDGE TIMEOUT after {max_retries} retries: {str(last_err)[:60]}",
    }


def print_colored(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")


def score_color(score):
    if score >= 4.5: return "92"   # bright green
    if score >= 3.5: return "32"   # green
    if score >= 2.5: return "33"   # yellow
    return "31"                     # red


def score_bar(score, max_score=5):
    filled = int(score * 4)  # 20 chars for 5 points
    empty = 20 - filled
    return f"[{'█' * filled}{'░' * empty}] {score:.1f}/5"


def main():
    global MIRA_URL

    parser = argparse.ArgumentParser(description="Mira Tutor Eval — Quality Metrics")
    parser.add_argument(
        "--target", choices=list(TARGETS.keys()), default=None,
        help="Deployment target (local, oss, elevenlabs). Overrides MIRA_URL.",
    )
    args = parser.parse_args()

    if args.target:
        target = TARGETS[args.target]
        MIRA_URL = target["http_url"]
        target_name = f"{args.target} — {target['description']}"
    else:
        target_name = "custom" if MIRA_URL != "http://localhost:7860" else "local"

    if not JUDGE_API_KEY or JUDGE_API_KEY == "DUMMY_KEY":
        print("ERROR: Set LLM_API_KEY or OPENAI_API_KEY for the judge model")
        sys.exit(1)

    auth_status = "JWT ✓" if WEBUI_SECRET_KEY else "no auth"

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          MIRA TUTOR EVAL — Judge Model Assessment           ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Target: {target_name:<50} ║")
    print(f"║  Mira:   {MIRA_URL:<50} ║")
    print(f"║  Auth:   {auth_status:<50} ║")
    print(f"║  Judge:  {JUDGE_MODEL:<50} ║")
    print(f"║  Tests:  {len(TEST_CASES):<50} ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    results = []
    category_scores = {}

    skipped = 0
    for i, test in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] {test.label}...", end=" ", flush=True)

        try:
            # Get Mira's response
            mira = get_mira_response(test)

            if mira["content"].startswith("[ERROR"):
                print_colored(f"SKIP — {mira['content'][:80]}", "\033[33m")
                skipped += 1
                continue

            # Judge it
            scores = judge_response(test, mira["content"])
        except Exception as e:
            print_colored(f"SKIP — {type(e).__name__}: {str(e)[:60]}", "\033[33m")
            skipped += 1
            continue

        results.append({
            "test": test,
            "mira_response": mira["content"],
            "tokens": mira["tokens"],
            "latency_ms": mira["latency_ms"],
            "scores": scores,
        })

        # Accumulate category scores
        cat = test.category
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(scores)

        overall = scores.get("overall", 0)
        print_colored(f"{'★' * overall}{'☆' * (5-overall)} ({overall}/5) — {scores.get('one_line_feedback', '')[:60]}", score_color(overall))

    if skipped:
        print(f"\n⚠️  {skipped}/{len(TEST_CASES)} tests skipped due to errors")

    # ── Detailed Results ──
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    for r in results:
        test = r["test"]
        s = r["scores"]
        print(f"\n{'─' * 70}")
        print(f"📋 {test.id} | {test.label} | {test.category}")
        print(f"👤 {test.user_name or '?'} | 📚 {test.topic or 'none'}")
        print(f"💬 USER: {test.user_msg[:80]}")
        print(f"🤖 MIRA ({r['tokens']}tok, {r['latency_ms']}ms): {r['mira_response'][:150]}...")
        print(f"   Friendly:    {score_bar(s.get('friendly_warm', 0))}")
        print(f"   Educational: {score_bar(s.get('educational_encouraging', 0))}")
        print(f"   Indian:      {score_bar(s.get('indian_cultural', 0))}")
        print(f"   EQ:          {score_bar(s.get('emotional_intelligence', 0))}")
        print(f"   Socratic:    {score_bar(s.get('socratic_teaching', 0))}")
        print(f"   Brevity:     {score_bar(s.get('brevity_voice_ready', 0))}")
        print(f"   💡 {s.get('one_line_feedback', '')}")

    # ── Category Averages ──
    print("\n" + "=" * 70)
    print("CATEGORY AVERAGES")
    print("=" * 70)

    dims = ["friendly_warm", "educational_encouraging", "indian_cultural",
            "emotional_intelligence", "socratic_teaching", "brevity_voice_ready", "overall"]

    for cat, score_list in sorted(category_scores.items()):
        print(f"\n📁 {cat.upper()} ({len(score_list)} tests)")
        for dim in dims:
            vals = [s.get(dim, 0) for s in score_list]
            avg = sum(vals) / len(vals) if vals else 0
            label = dim.replace("_", " ").title()
            print(f"   {label:<25} {score_bar(avg)}")

    # ── Overall Summary ──
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    all_scores = [r["scores"] for r in results]
    for dim in dims:
        vals = [s.get(dim, 0) for s in all_scores]
        avg = sum(vals) / len(vals) if vals else 0
        label = dim.replace("_", " ").title()
        color = score_color(avg)
        print_colored(f"   {label:<25} {score_bar(avg)}", color)

    # Average latency and tokens
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    print(f"\n   Avg latency: {avg_latency:.0f}ms")
    print(f"   Avg tokens:  {avg_tokens:.0f}")

    # ── Weakest Areas ──
    print("\n" + "=" * 70)
    print("TOP IMPROVEMENT AREAS")
    print("=" * 70)

    dim_avgs = {}
    for dim in dims[:-1]:  # exclude overall
        vals = [s.get(dim, 0) for s in all_scores]
        dim_avgs[dim] = sum(vals) / len(vals)

    for dim, avg in sorted(dim_avgs.items(), key=lambda x: x[1]):
        label = dim.replace("_", " ").title()
        if avg < 4.0:
            print_colored(f"   ⚠️  {label}: {avg:.1f}/5 — needs work", "33")
        else:
            print_colored(f"   ✅ {label}: {avg:.1f}/5 — good", "32")

    # ── Worst Individual Scores ──
    print(f"\n{'─' * 70}")
    print("LOWEST SCORING RESPONSES (overall < 4):")
    low = [r for r in results if r["scores"].get("overall", 0) < 4]
    if not low:
        print("   None! All responses scored 4+")
    for r in sorted(low, key=lambda x: x["scores"].get("overall", 0)):
        s = r["scores"]
        print(f"   • {r['test'].id} ({s.get('overall',0)}/5): {s.get('one_line_feedback','')}")

    # ── Save full results to JSON ──
    target_suffix = args.target or "local"
    output_path = f"tests/eval_results_{target_suffix}.json"
    json_results = []
    for r in results:
        json_results.append({
            "id": r["test"].id,
            "label": r["test"].label,
            "category": r["test"].category,
            "user_msg": r["test"].user_msg,
            "user_name": r["test"].user_name,
            "topic": r["test"].topic,
            "mira_response": r["mira_response"],
            "tokens": r["tokens"],
            "latency_ms": r["latency_ms"],
            "scores": r["scores"],
        })

    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "target": target_suffix,
            "mira_url": MIRA_URL,
            "judge_model": JUDGE_MODEL,
            "results": json_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Full results saved to {output_path}")

    print("\n🏁 EVAL COMPLETE!")


if __name__ == "__main__":
    main()
