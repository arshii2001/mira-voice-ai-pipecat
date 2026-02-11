"""
Curriculum Manager — loads structured curriculum JSON files and provides
token-budgeted context blocks for the LLM system prompt.

Scans CURRICULUM_DIR at startup, indexes chapters/sections/concepts/text_blocks
for O(1) lookup. When a room topic is set, builds a rich context block with:
  - Concept summary + related concepts
  - NCERT source text (text blocks)
  - FAQs and follow-up questions
  - Chapter/section hierarchy

Token budget is configurable via CURRICULUM_CONTEXT_MAX_TOKENS env var.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
CURRICULUM_DIR = os.environ.get("CURRICULUM_DIR", "/content/curriculum")
CURRICULUM_CONTEXT_MAX_TOKENS = int(
    os.environ.get("CURRICULUM_CONTEXT_MAX_TOKENS", "600")
)

# Rough token estimate: 1 token ≈ 4 chars for English
CHARS_PER_TOKEN = 4


class CurriculumManager:
    """Loads and indexes curriculum JSON files for fast topic lookup."""

    def __init__(self, curriculum_dir: Optional[str] = None):
        self._dir = curriculum_dir or CURRICULUM_DIR
        # Indexed data
        self._files: dict[str, dict] = {}           # filename -> raw JSON
        self._concepts: dict[str, dict] = {}         # concept_name (lower) -> concept entry
        self._concept_file: dict[str, str] = {}      # concept_name (lower) -> filename
        self._chapters: dict[str, dict] = {}         # chapter_id -> chapter entry
        self._chapter_file: dict[str, str] = {}      # chapter_id -> filename
        self._sections: dict[str, dict] = {}         # section_id -> section entry
        self._section_chapter: dict[str, str] = {}   # section_id -> chapter_id
        self._text_blocks: dict[str, dict] = {}      # text_block_id -> text block entry
        self._block_section: dict[str, str] = {}     # text_block_id -> section_id
        # Browseable tree: filename -> [{chapter_id, title, sections: [{section_id, title, concepts}]}]
        self._trees: dict[str, list] = {}

        self._load_all()

    # ── Loading & Indexing ────────────────────────────────────────────

    def _load_all(self):
        """Scan curriculum directory and index all JSON files."""
        curriculum_path = Path(self._dir)
        if not curriculum_path.exists():
            logger.warning(
                f"[CURRICULUM] Directory not found: {self._dir} — curriculum features disabled"
            )
            return

        json_files = list(curriculum_path.glob("*.json"))
        if not json_files:
            logger.warning(
                f"[CURRICULUM] No JSON files in {self._dir} — curriculum features disabled"
            )
            return

        for fpath in json_files:
            try:
                self._load_file(fpath)
            except Exception as e:
                logger.error(f"[CURRICULUM] Failed to load {fpath.name}: {e}")

        logger.info(
            f"[CURRICULUM] Loaded {len(self._files)} file(s): "
            f"{len(self._concepts)} concepts, "
            f"{len(self._chapters)} chapters, "
            f"{len(self._text_blocks)} text blocks"
        )

    def _load_file(self, fpath: Path):
        """Load and index a single curriculum JSON file."""
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        fname = fpath.name
        self._files[fname] = data

        # Index concept registry
        concept_registry = data.get("concept_registry", {})
        for cname, centry in concept_registry.items():
            key = cname.lower()
            self._concepts[key] = centry
            self._concept_file[key] = fname

        # Index textbook hierarchy
        tree = []
        for chapter in data.get("textbook_hierarchy", []):
            cid = chapter["chapter_id"]
            self._chapters[cid] = chapter
            self._chapter_file[cid] = fname

            chapter_tree = {
                "chapter_id": cid,
                "title": chapter.get("title", cid),
                "sections": [],
            }

            for section in chapter.get("sections", []):
                sid = section["section_id"]
                self._sections[sid] = section
                self._section_chapter[sid] = cid

                section_tree = {
                    "section_id": sid,
                    "title": section.get("title", sid),
                    "concepts": section.get("concepts", []),
                }
                chapter_tree["sections"].append(section_tree)

                for block in section.get("text_blocks", []):
                    bid = block["text_block_id"]
                    self._text_blocks[bid] = block
                    self._block_section[bid] = sid

            tree.append(chapter_tree)

        self._trees[fname] = tree

    # ── Public API ────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if at least one curriculum file is loaded."""
        return len(self._files) > 0

    def list_files(self) -> list[dict]:
        """Return list of loaded curriculum files with metadata."""
        result = []
        for fname, data in self._files.items():
            meta = data.get("metadata", {})
            result.append({
                "filename": fname,
                "source": meta.get("source", fname),
                "total_concepts": meta.get("total_concepts", 0),
                "total_text_blocks": meta.get("total_text_blocks", 0),
                "languages": meta.get("languages", []),
            })
        return result

    def get_browseable_tree(self, filename: Optional[str] = None) -> list[dict]:
        """Return the chapter → section → concepts tree for the topic picker.

        If filename is None, returns trees from all files merged.
        """
        if filename and filename in self._trees:
            return self._trees[filename]

        # Merge all trees
        merged = []
        for tree in self._trees.values():
            merged.extend(tree)
        return merged

    def search_concepts(self, query: str, limit: int = 20) -> list[dict]:
        """Search concepts by name (case-insensitive substring match)."""
        q = query.lower()
        results = []
        for key, centry in self._concepts.items():
            if q in key:
                results.append({
                    "concept_id": centry.get("concept_id", key),
                    "concept_name": centry.get("concept_name", key),
                    "chapter_ids": centry.get("chapter_ids", []),
                    "occurrences": centry.get("occurrences", 0),
                })
                if len(results) >= limit:
                    break
        return results

    def get_context_for_topic(
        self,
        topic: str,
        max_tokens: Optional[int] = None,
        language: str = "english",
    ) -> Optional[str]:
        """Build a curriculum context block for the given topic.

        Returns a formatted string to inject into the system prompt,
        or None if the topic is not found in any curriculum.

        Priority for filling token budget:
          1. Concept summary + related concepts (~100 tokens)
          2. Chapter/section info (~30 tokens)
          3. FAQs (~150 tokens)
          4. NCERT source text blocks (fill remaining budget)
        """
        budget = max_tokens or CURRICULUM_CONTEXT_MAX_TOKENS
        budget_chars = budget * CHARS_PER_TOKEN

        # Look up concept
        key = topic.lower()
        concept = self._concepts.get(key)
        if not concept:
            # Try fuzzy: check if topic is a substring of any concept
            for ckey, centry in self._concepts.items():
                if key in ckey or ckey in key:
                    concept = centry
                    key = ckey
                    break

        if not concept:
            return None

        lines: list[str] = []
        used_chars = 0

        # ── 1. Concept header + summary ──────────────────────────────
        concept_name = concept.get("concept_name", topic)
        lines.append(f"Topic: {concept_name}")

        # Find chapter and section titles
        chapter_ids = concept.get("chapter_ids", [])
        section_ids = concept.get("section_ids", [])

        if chapter_ids:
            ch = self._chapters.get(chapter_ids[0])
            if ch:
                lines.append(f"Source: {ch.get('title', chapter_ids[0])}")

        if section_ids:
            sec = self._sections.get(section_ids[0])
            if sec:
                lines.append(f"Section: {sec.get('title', section_ids[0])}")

        # Related concepts
        parents = concept.get("parents", [])
        children = concept.get("children", [])
        siblings = concept.get("siblings", [])
        related = concept.get("related_concepts", [])

        rel_parts = []
        if parents:
            rel_parts.append(f"Part of: {', '.join(parents)}")
        if children:
            rel_parts.append(f"Subtopics: {', '.join(children)}")
        if siblings:
            rel_parts.append(f"Related: {', '.join(siblings)}")
        if related:
            high_conf = [
                r for r in related if r.get("confidence", 0) >= 0.8
            ]
            if high_conf:
                rel_strs = [
                    f"{r['target_concept']} ({r['relationship']})"
                    for r in high_conf[:4]
                ]
                rel_parts.append(f"Connections: {', '.join(rel_strs)}")

        if rel_parts:
            lines.append("")
            lines.extend(rel_parts)

        header_text = "\n".join(lines)
        used_chars += len(header_text)

        # ── 2. Section summary ───────────────────────────────────────
        summary_key = f"summary_{language}" if language != "english" else "summary_english"
        if section_ids:
            sec = self._sections.get(section_ids[0])
            if sec:
                sec_summary = sec.get("summary_english", "")
                if sec_summary and used_chars + len(sec_summary) + 20 < budget_chars:
                    lines.append("")
                    lines.append(f"Overview: {sec_summary}")
                    used_chars += len(sec_summary) + 20

        # ── 3. FAQs ─────────────────────────────────────────────────
        faqs_added = 0
        if chapter_ids:
            ch = self._chapters.get(chapter_ids[0])
            if ch:
                faqs = ch.get("faqs", [])
                # Filter FAQs relevant to this concept
                relevant_faqs = [
                    faq for faq in faqs
                    if key in faq.get("question", "").lower()
                    or key in faq.get("answer", "").lower()
                ]
                if not relevant_faqs:
                    relevant_faqs = faqs[:2]  # fallback: first 2

                if relevant_faqs and used_chars + 30 < budget_chars:
                    lines.append("")
                    lines.append("Key Questions:")
                    for faq in relevant_faqs[:3]:
                        faq_text = f"Q: {faq['question']}\nA: {faq['answer']}"
                        if used_chars + len(faq_text) + 10 < budget_chars:
                            lines.append(faq_text)
                            used_chars += len(faq_text) + 10
                            faqs_added += 1
                        else:
                            break

        # ── 4. NCERT source text blocks ──────────────────────────────
        block_ids = concept.get("text_block_ids", [])
        if block_ids and used_chars + 50 < budget_chars:
            lines.append("")
            lines.append("NCERT Reference Text:")
            for bid in block_ids:
                block = self._text_blocks.get(bid)
                if not block:
                    continue

                # Prefer the language-specific summary, fall back to original
                block_text = block.get(summary_key) or block.get(
                    "summary_english"
                )
                if not block_text:
                    # Use original text, but truncate if needed
                    block_text = block.get("original_text", "")

                if not block_text:
                    continue

                remaining = budget_chars - used_chars - 20
                if remaining <= 50:
                    break

                if len(block_text) > remaining:
                    block_text = block_text[:remaining].rsplit(" ", 1)[0] + "..."

                lines.append(f'"{block_text}"')
                used_chars += len(block_text) + 10

        # ── 5. Follow-up suggestions ─────────────────────────────────
        if chapter_ids and used_chars + 80 < budget_chars:
            ch = self._chapters.get(chapter_ids[0])
            if ch:
                followups = ch.get("followups", [])
                if followups:
                    lines.append("")
                    lines.append(
                        "Suggested follow-ups: "
                        + " | ".join(followups[:3])
                    )

        return "\n".join(lines)

    def get_chapter_faqs(self, chapter_id: str) -> list[dict]:
        """Return FAQs for a specific chapter."""
        ch = self._chapters.get(chapter_id)
        if not ch:
            return []
        return ch.get("faqs", [])

    def get_concept_detail(self, concept_name: str) -> Optional[dict]:
        """Return full concept registry entry."""
        return self._concepts.get(concept_name.lower())

    def get_text_block(self, block_id: str) -> Optional[dict]:
        """Return a specific text block by ID."""
        return self._text_blocks.get(block_id)

    def get_section_context(
        self,
        section_id: str,
        max_tokens: Optional[int] = None,
        language: str = "english",
    ) -> Optional[str]:
        """Build curriculum context for a section (used by chapter:section dropdowns).

        Returns a formatted string for the system prompt, or None if section not found.
        Includes: section title, chapter title, concepts, text blocks, FAQs.
        """
        section = self._sections.get(section_id)
        if not section:
            return None

        budget = max_tokens or CURRICULUM_CONTEXT_MAX_TOKENS
        budget_chars = budget * CHARS_PER_TOKEN

        chapter_id = self._section_chapter.get(section_id)
        chapter = self._chapters.get(chapter_id) if chapter_id else None

        lines: list[str] = []
        used_chars = 0

        # Header
        sec_title = section.get("title", section_id)
        lines.append(f"Section: {sec_title}")
        if chapter:
            lines.append(f"Chapter: {chapter.get('title', chapter_id)}")

        # Concepts in this section
        concepts = section.get("concepts", [])
        if concepts:
            lines.append(f"Concepts covered: {', '.join(concepts)}")

        # Section summary
        summary_key = f"summary_{language}" if language != "english" else "summary_english"
        sec_summary = section.get(summary_key) or section.get("summary_english", "")
        if sec_summary:
            lines.append("")
            lines.append(f"Overview: {sec_summary}")

        used_chars = sum(len(l) for l in lines)

        # Text blocks in this section
        text_blocks = section.get("text_blocks", [])
        if text_blocks and used_chars + 50 < budget_chars:
            lines.append("")
            lines.append("NCERT Reference Text:")
            for block in text_blocks:
                block_text = block.get(summary_key) or block.get("summary_english") or block.get("original_text", "")
                if not block_text:
                    continue
                remaining = budget_chars - used_chars - 20
                if remaining <= 50:
                    break
                if len(block_text) > remaining:
                    block_text = block_text[:remaining].rsplit(" ", 1)[0] + "..."
                lines.append(f'"{block_text}"')
                used_chars += len(block_text) + 10

        # FAQs from chapter (filtered to section concepts)
        if chapter and used_chars + 50 < budget_chars:
            faqs = chapter.get("faqs", [])
            concept_keys = {c.lower() for c in concepts}
            relevant_faqs = [
                faq for faq in faqs
                if any(ck in faq.get("question", "").lower() or ck in faq.get("answer", "").lower() for ck in concept_keys)
            ]
            if not relevant_faqs:
                relevant_faqs = faqs[:2]
            if relevant_faqs:
                lines.append("")
                lines.append("Key Questions:")
                for faq in relevant_faqs[:3]:
                    faq_text = f"Q: {faq['question']}\nA: {faq['answer']}"
                    if used_chars + len(faq_text) + 10 < budget_chars:
                        lines.append(faq_text)
                        used_chars += len(faq_text) + 10
                    else:
                        break

        return "\n".join(lines)

    def get_section_info(self, section_id: str) -> Optional[dict]:
        """Return section title, chapter title, and concepts for display."""
        section = self._sections.get(section_id)
        if not section:
            return None
        chapter_id = self._section_chapter.get(section_id)
        chapter = self._chapters.get(chapter_id) if chapter_id else None
        return {
            "section_id": section_id,
            "section_title": section.get("title", section_id),
            "chapter_id": chapter_id,
            "chapter_title": chapter.get("title", chapter_id) if chapter else None,
            "concepts": section.get("concepts", []),
        }


# ── Singleton instance ────────────────────────────────────────────────
_instance: Optional[CurriculumManager] = None


def get_curriculum_manager() -> CurriculumManager:
    """Return the singleton CurriculumManager, creating it on first call."""
    global _instance
    if _instance is None:
        _instance = CurriculumManager()
    return _instance
