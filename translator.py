"""
Classroom Translator — translates text between languages using the LLM.

Provides a lightweight translation service for the classroom broadcast system.
Uses the same OpenAI-compatible LLM already configured in bot.py.

Usage:
    translator = Translator(api_key="...", base_url="...", model="gpt-4o-mini")
    result = await translator.translate("Hello, how are you?", target_lang="hi")
    # result = "नमस्ते, तुम कैसे हो?"
"""

import logging
import time
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Language code → full name mapping
LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
}

TRANSLATION_SYSTEM_PROMPT = """You are a real-time translator for a multilingual classroom.
Your ONLY job is to translate the given text to {target_language}.

Rules:
- Output ONLY the translation. No explanations, no notes, no prefixes.
- Preserve the tone and style — if it's casual, keep it casual.
- If the text is already in the target language, return it unchanged.
- Hindi must be in Devanagari script, Tamil in Tamil script.
- Keep it natural and spoken — this will be read aloud by TTS.
- Do NOT add quotes or attribution.
- Remove any [User is speaking ...] tags from the input before translating.
"""


class Translator:
    """Translates text between languages using the configured LLM."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._base_url = base_url

        # Session-level metrics
        self._call_count: int = 0
        self._total_latency_ms: float = 0.0
        self._skipped_count: int = 0

        logger.info(f"Translator initialized: model={model}, base_url={base_url}")

    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None,
    ) -> str:
        """
        Translate text to the target language.

        Args:
            text: Text to translate
            target_lang: Target language code (en, hi, ta)
            source_lang: Optional source language code (for logging)

        Returns:
            Translated text string
        """
        target_name = LANG_NAMES.get(target_lang, target_lang)

        # Skip translation if source and target are the same
        if source_lang and source_lang == target_lang:
            self._skipped_count += 1
            logger.debug(f"Skipping translation: source={source_lang} == target={target_lang}")
            return text

        system_prompt = TRANSLATION_SYSTEM_PROMPT.format(target_language=target_name)
        t0 = time.monotonic()

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=500,
                temperature=0.3,  # Low temperature for faithful translation
                **({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if "api.openai.com" not in self._base_url else {}),
            )

            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            raw_content = response.choices[0].message.content
            if raw_content is None:
                # Reasoning model may return content=None with reasoning_content only
                logger.warning(
                    f"[TRANSLATE] LLM returned content=None for {source_lang}→{target_lang}, "
                    f"falling back to original text"
                )
                return text
            translated = raw_content.strip()

            # Update session metrics
            self._call_count += 1
            self._total_latency_ms += latency_ms
            avg_ms = round(self._total_latency_ms / self._call_count, 1)

            # Token usage
            usage = response.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0

            logger.info(
                f"[METRICS][TRANSLATE] {source_lang or '?'} → {target_lang} | "
                f"latency={latency_ms}ms | tokens={tokens_in}+{tokens_out} | "
                f"avg={avg_ms}ms (n={self._call_count}) | "
                f"'{text[:50]}' → '{translated[:50]}'"
            )
            return translated

        except Exception as e:
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            logger.error(f"[METRICS][TRANSLATE] FAILED {source_lang or '?'} → {target_lang} | latency={latency_ms}ms | error={e}")
            # Fallback: return original text
            return text
