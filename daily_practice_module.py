# ============================================================
# daily_practice_module.py
# Updated: replaced static PHRASES list with a dynamic LLM call
# so every time the daily practice button is toggled visible,
# a fresh Mandarin phrase is generated with full pinyin breakdown
# ============================================================

import json

# ------------------------------------------------------------
# 1. LLM-generated daily phrase
# Called each time toggle_daily_phrase() makes the panel visible
# Replaces the old static PHRASES list + date-index lookup
# ------------------------------------------------------------
def get_daily_phrase_from_llm(client, model: str) -> dict:
    """
    Asks the LLM to generate a random, practical Mandarin phrase
    with full pinyin and per-character pronunciation breakdown.
    Returns a structured dict matching the old PHRASES format
    so the rest of the module works unchanged.
    """

    prompt = """
Generate a single practical Mandarin Chinese phrase for a language learner.
Choose a random topic from: greetings, food, travel, family, weather, shopping, emotions, or daily life.
Vary the difficulty — sometimes simple (2-3 chars), sometimes moderate (4-6 chars).

Return ONLY valid JSON in this exact schema, no extra text:

{
  "mandarin": "<Chinese characters>",
  "english": "<English translation>",
  "pinyin": ["<pinyin for each character>"],
  "breakdown": [
    {
      "char": "<single character>",
      "pinyin": "<pinyin>",
      "guide": "sounds like '<english sound>' with a <tone name> (<tone number>)"
    }
  ],
  "topic": "<topic category used>"
}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},  # enforce structured JSON response
    )

    # Parse the JSON response into a dict
    phrase_info = json.loads(response.choices[0].message.content)
    return phrase_info


# ------------------------------------------------------------
# 2. Build readable text output for UI
# Unchanged from original — works with both old and new phrase format
# ------------------------------------------------------------
def build_daily_phrase_text(phrase_info: dict) -> str:
    """Return formatted text with Mandarin, translation, breakdown, and practice tips."""

    breakdown_text = "\n".join(
        f"• {b['char']} ({b['pinyin']}) — {b['guide']}"
        for b in phrase_info.get("breakdown", [])
    )

    # Include topic label if the LLM provided one
    topic_line = f"Topic: {phrase_info['topic']}\n" if "topic" in phrase_info else ""

    return f"""
DAILY PRACTICE PHRASE

{topic_line}Mandarin: {phrase_info['mandarin']}
Meaning: {phrase_info['english']}
Pinyin: {' '.join(phrase_info['pinyin'])}

Pronunciation Breakdown:
{breakdown_text}

Practice Tips:
- Say it slowly first
- Focus on tones
- Repeat 5 times out loud
- Then record yourself below
""".strip()


# ------------------------------------------------------------
# 3. Speak the phrase using the existing TTS function
# Unchanged from original
# ------------------------------------------------------------
def speak_daily_phrase(phrase_info: dict, talker_fn, voice: str):
    """Return audio bytes of the Mandarin phrase spoken aloud."""
    return talker_fn(phrase_info["mandarin"], voice)


# ------------------------------------------------------------
# 4. Single pipeline entry point for Gradio
# Updated signature: now requires client + model to call LLM
# dadbot.py passes these in from its own initialized client/MODEL
# ------------------------------------------------------------
def daily_phrase_pipeline(voice: str, talker_fn, client, model: str):
    """
    Returns:
        audio_output: bytes (TTS of the Mandarin phrase)
        text_output: str (formatted phrase with full breakdown)

    Changed from original: client and model are now required params
    so this module doesn't need its own OpenAI initialization.
    """
    # Generate a fresh phrase from the LLM every time this is called
    phrase_info = get_daily_phrase_from_llm(client, model)
    text_output = build_daily_phrase_text(phrase_info)
    audio_output = speak_daily_phrase(phrase_info, talker_fn, voice)
    return audio_output, text_output