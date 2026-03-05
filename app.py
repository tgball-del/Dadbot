# ============================================================
# app.py — Dad's Babelbot
# Mandarin language learning assistant with:
#   - AI chat with pinyin enforcement
#   - Daily phrase practice with pronunciation scoring
#   - Character drawing pad with vision recognition
#   - Per-user progress tracking
# ============================================================

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import os
import json
import time
import threading
import re
import base64
import io
import numpy as np
import pandas as pd

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from progress_module import save_session, build_progress_display, empty_trend_df
from auth_module import register_user, verify_login
from daily_practice_module import get_daily_phrase_from_llm, build_daily_phrase_text, speak_daily_phrase

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')  # use secret on HuggingFace Space
MODEL = "gpt-4.1-mini"
client = OpenAI()

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
# MALE_VOICES = ["ash", "ballad", "echo", "onyx"]
# added all voices not just male voices (for dadbot) to dropdown
MALE_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"]

image_address = (
    "https://static.vecteezy.com/system/resources/previews/014/477/240/original/"
    "chinese-words-cute-girl-saying-hello-in-chinese-language-learning-chinese-"
    "language-isolated-illustration-vector.jpg"
)

example_msgs = [
        "How do you say 'I love you' in Mandarin Chinese?"
]

# ------------------------------------------------------------
# System prompt
# ------------------------------------------------------------
system_prompt = """
You are a language expert. Primarily Mandarin Chinese, but only translate when requested.
Always be accurate. Pronunciation is critical.
Occasionally include the Mandarin word of the day for a random subject (animal, food, common object, weather, etc...).

=== MANDATORY PINYIN RULE ===
For EVERY Mandarin word or phrase you use — no exceptions — you MUST include ALL of the following:
1. The Chinese characters
2. The full pinyin string in parentheses
3. A per-character breakdown with tone description

Use this EXACT format for every Mandarin word or phrase:

Chinese: 我爱你 (wǒ ài nǐ)
Breakdown:
- 我 (wǒ) — sounds like "waw" — falling-rising tone (3rd tone)
- 爱 (ài) — sounds like "eye" — falling tone (4th tone)
- 你 (nǐ) — sounds like "nee" — falling-rising tone (3rd tone)

This breakdown is REQUIRED every single time Chinese characters appear in your response.
Even for single characters, you must show the pinyin and tone description.
=== END MANDATORY PINYIN RULE ===
"""

# ============================================================
# RETRY / TIMEOUT UTILITIES
# ============================================================

class TimeoutException(Exception):
    pass


def retry_api_call(func, max_attempts=3, delay=1):
    """
    Generic retry wrapper for unstable external API calls.
    Uses exponential backoff between retries.
    Raises the final exception if all attempts fail.
    """
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (2 ** attempt))
    raise last_exception


def run_with_timeout(func, timeout=25):
    """
    Cross-platform timeout protection using a daemon thread.
    Prevents HuggingFace Space workers from hanging indefinitely
    during external API calls. Default timeout = 25 seconds.
    """
    result_container = {}

    def worker():
        try:
            result_container["result"] = func()
        except Exception as e:
            result_container["error"] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutException("Operation timed out")

    if "error" in result_container:
        raise result_container["error"]

    return result_container.get("result")


# ============================================================
# RESPONSE CACHE
# Reduces latency for repeated or similar chat prompts.
# ============================================================
response_cache = {}


# ============================================================
# VOICE / TTS
# ============================================================

def talker(message, voice):
    """Convert text to speech using OpenAI TTS. Returns audio bytes."""
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=message,
        instructions="Use a normal conversational pace and tone",
        speed=1.25,
        response_format="wav"   # wav required for iPhone Safari compatibility
    )
    return response.content


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe_audio(audio_file: str, prompt_hint: str = "") -> str:
    """
    Transcribe recorded Mandarin speech to text using Whisper.
    prompt_hint: optional target phrase to steer character selection
    and prevent homophone hallucination.
    Wrapped in retry + timeout for HuggingFace Space stability.
    """
    if not audio_file:
        return "Transcription error: no audio file provided."

    try:
        def api_call():
            def transcription_task():
                with open(audio_file, "rb") as f:
                    return client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=f,
                        prompt=prompt_hint if prompt_hint else None,
                        language="zh"
                    ).text
            return retry_api_call(transcription_task)

        return run_with_timeout(api_call, timeout=25)

    except TimeoutException:
        return "Transcription error: request timed out."
    except Exception as e:
        return f"Transcription error: {str(e)}"


# ============================================================
# PINYIN ENFORCEMENT MODULE
# ============================================================

def contains_chinese(text: str) -> bool:
    """Returns True if text contains any CJK Unified Ideograph characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def extract_chinese_phrases(text: str) -> list:
    """
    Extracts all contiguous sequences of Chinese characters from text.
    Returns a deduplicated list preserving order of first appearance.
    """
    phrases = re.findall(r'[\u4e00-\u9fff]+', text)
    seen = set()
    unique_phrases = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique_phrases.append(p)
    return unique_phrases


def generate_pinyin_breakdown(phrases: list) -> str:
    """
    Calls the model to generate a structured pinyin + tone breakdown
    for each Chinese phrase. Returns a formatted string block.
    """
    if not phrases:
        return ""

    phrase_list = "\n".join(f"- {p}" for p in phrases)
    breakdown_prompt = f"""
For each of the following Mandarin Chinese phrases or characters, provide a structured pronunciation guide.

Phrases:
{phrase_list}

For EACH phrase use this exact format:

Chinese: <characters> (<full pinyin>)
Breakdown:
- <char> (<pinyin>) — sounds like "<english sound guide>" — <tone name> (<tone number>)
(one bullet per character)

Return ONLY the formatted breakdowns, no extra commentary.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": breakdown_prompt}]
    )
    return response.choices[0].message.content.strip()


def enforce_pinyin_in_reply(reply: str) -> str:
    """
    Safety net applied to every assistant reply.
    If Chinese characters are present and the model did not already
    include a full tone-marked breakdown, one is appended.
    """
    if not contains_chinese(reply):
        return reply

    phrases = extract_chinese_phrases(reply)

    has_tone_marks = bool(re.search(r'[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]', reply))
    has_breakdown_bullets = '•' in reply or '·' in reply

    if has_tone_marks and has_breakdown_bullets:
        return reply

    breakdown = generate_pinyin_breakdown(phrases)
    if breakdown:
        reply += f"\n\n---\n📖 **Pronunciation Guide**\n{breakdown}"

    return reply


# ============================================================
# MANDARIN SCORING MODULE
# ============================================================

def evaluate_mandarin(transcript: str) -> dict:
    """
    Professional Mandarin pronunciation evaluation.
    Returns structured rubric scores as a dict.
    """
    rubric_prompt = f"""
You are a professional Mandarin pronunciation examiner.

Student transcript:
{transcript}

Evaluate across FIVE categories:

1. tone_accuracy
2. pronunciation_clarity
3. fluency
4. grammar
5. overall_score

Scoring rules:
- Each numeric score must be 0–100.
- overall_score must reflect the category scores.
- Be realistic but encouraging.

Return ONLY valid JSON in this exact schema:

{{
  "tone_accuracy": int,
  "pronunciation_clarity": int,
  "fluency": int,
  "grammar": int,
  "overall_score": int,
  "strengths": "short sentence",
  "improvements": "short sentence",
  "coach_feedback": "encouraging 1–2 sentence coaching message"
}}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": rubric_prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def format_feedback(transcript: str, scores: dict) -> str:
    """Format rubric scores into clean coaching text for UI display."""
    return f"""
TRANSCRIPT
{transcript}

SCORES
Tone Accuracy: {scores['tone_accuracy']}
Pronunciation Clarity: {scores['pronunciation_clarity']}
Fluency: {scores['fluency']}
Grammar: {scores['grammar']}
Overall: {scores['overall_score']}

STRENGTHS
{scores['strengths']}

IMPROVEMENTS
{scores['improvements']}

COACH
{scores['coach_feedback']}
""".strip()


def score_audio_pipeline(audio_file, voice, target_phrase=None, username=None):
    """
    End-to-end scoring pipeline:
    audio → transcript → rubric scores → formatted feedback → TTS

    target_phrase: dict from daily_phrase_state (None for free practice)
    username:      logged-in username for progress tracking
    """
    if audio_file is None:
        return None, "No audio provided."

    transcription_hint = ""
    if target_phrase:
        transcription_hint = (
            f"The speaker is practicing this Mandarin phrase: "
            f"{target_phrase['mandarin']} "
            f"({' '.join(target_phrase['pinyin'])})"
        )

    transcript = transcribe_audio(audio_file, prompt_hint=transcription_hint)
    scores = evaluate_mandarin(transcript)
    feedback_text = format_feedback(transcript, scores)

    if target_phrase:
        breakdown_lines = "\n".join(
            f"• {b['char']} ({b['pinyin']}) — {b['guide']}"
            for b in target_phrase.get("breakdown", [])
        )
        pinyin_str = ' '.join(target_phrase['pinyin'])
        reference_block = (
            f"Target Phrase: {target_phrase['mandarin']} ({pinyin_str})\n"
            f"Meaning: {target_phrase['english']}\n\n"
            f"Pronunciation Reference:\n{breakdown_lines}"
        )
    else:
        reference_block = generate_pinyin_breakdown(
            extract_chinese_phrases(transcript)
        )

    if reference_block:
        feedback_text += f"\n\n---\n📖 PRONUNCIATION REFERENCE\n{reference_block}"

    if username:
        phrase_mandarin = target_phrase["mandarin"] if target_phrase else transcript
        phrase_english  = target_phrase["english"]  if target_phrase else "(free practice)"
        save_session(username, phrase_mandarin, phrase_english, scores)

    spoken_feedback = talker(scores["coach_feedback"], voice)
    return spoken_feedback, feedback_text


# ============================================================
# CHAT
# ============================================================

def put_message_in_chatbot(message, history):
    """Move user input into the chatbot history and clear the textbox."""
    return "", history + [{"role": "user", "content": message}]


def chat(history, voice):
    """
    Main chat function. Builds full message history with system prompt,
    checks response cache, runs inference if needed, enforces pinyin,
    then returns updated history and spoken audio.
    """
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_prompt}] + history

    prompt_text = "\n".join([m["content"] for m in messages])
    cache_key = f"{MODEL}:{prompt_text}"

    cached_reply = response_cache.get(cache_key)

    if cached_reply:
        reply = cached_reply
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        reply = response.choices[0].message.content
        response_cache[cache_key] = reply

    reply = enforce_pinyin_in_reply(reply)
    history += [{"role": "assistant", "content": reply}]
    speech = talker(reply, voice)

    return history, speech


# ============================================================
# DAILY PHRASE — two-step toggle for fast UI response
# ============================================================

def toggle_daily_phrase(state: bool):
    """
    Step 1: Instantly flips panel visibility and shows a placeholder.
    Does no LLM work — returns immediately for fast UI response.
    The .then() chain calls generate_daily_phrase() next.
    """
    new_state = not state

    if not new_state:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            False,
            None
        )

    return (
        gr.update(value=None, visible=True),
        gr.update(value="⏳ Generating your practice phrase...", visible=True),
        True,
        None
    )


def generate_daily_phrase(state: bool, voice: str):
    """
    Step 2: Does the real LLM + TTS work, called via .then() after toggle.
    If state is False (panel was just hidden), skips generation entirely.
    """
    if not state:
        return gr.update(), gr.update(), None

    try:
        phrase_info = get_daily_phrase_from_llm(client, MODEL)
        text = build_daily_phrase_text(phrase_info)
        audio = speak_daily_phrase(phrase_info, talker, voice)
    except Exception as e:
        phrase_info = None
        text = f"⚠️ Error generating phrase: {str(e)}"
        audio = None

    return (
        gr.update(value=audio),
        gr.update(value=text),
        phrase_info
    )


# ============================================================
# CHARACTER PAD
# ============================================================
def identify_drawn_character(image_input):
    """
    Receives a sketch dict from gr.Sketchpad.
    Uses the 'composite' key which contains the merged drawing as a numpy array.
    Converts to base64 PNG and sends to GPT-4o vision.
    Returns (None, analysis text) — audio explicitly cleared on each call.
    """
    if image_input is None:
        return None, "No drawing received. Please draw a character first."

    composite = image_input["composite"]

    pil_image = Image.fromarray(composite.astype(np.uint8))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    vision_prompt = """
You are a Mandarin Chinese expert examining a hand-drawn character.

Identify the Chinese character the user attempted to draw.
Even if the drawing is rough or imperfect, make your best identification.

Then provide the full structured breakdown in this EXACT format:

Character: <character>
Pinyin: <pinyin with tone marks>
Meaning: <English meaning>
Tone: <tone name and number>

Pronunciation Breakdown:
- <char> (<pinyin>) — sounds like "<english sound guide>" — <tone name> (<tone number>)

Stroke Count: <number>
Common Usage: <1-2 example words or phrases using this character>

Encouragement: <one sentence of encouragement for the learner>
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": vision_prompt
                    }
                ]
            }
        ],
        max_tokens=500
    )

    return None, response.choices[0].message.content.strip()

# ---------------- end identify_drawn_character ---------------
def lookup_character(character_input: str, voice: str):
    """
    Takes a typed or pasted Mandarin character or short phrase.
    Returns full pinyin breakdown, meaning, stroke info, and examples.
    Speaks the result aloud using talker().
    """
    if not character_input or character_input.strip() == "":
        return None, "Please enter a character or phrase to look up."

    character_input = character_input.strip()

    lookup_prompt = f"""
You are a Mandarin Chinese language expert.

The student wants to look up: {character_input}

Provide a complete structured reference in this EXACT format:

Character: {character_input}
Pinyin: <full pinyin with tone marks>
Meaning: <English meaning>

Pronunciation Breakdown:
- <char> (<pinyin>) — sounds like "<english sound guide>" — <tone name> (<tone number>)
(one bullet per character)

Stroke Count: <total strokes>
Radical: <radical name and meaning>
Tone: <tone name and number for each character>

Example Usage:
1. <example sentence in Chinese> (<pinyin>) — <English meaning>
2. <example sentence in Chinese> (<pinyin>) — <English meaning>

Memory Tip: <one creative tip to remember this character>
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": lookup_prompt}]
    )

    result_text = response.choices[0].message.content.strip()
    spoken = talker(result_text, voice)
    return spoken, result_text


# ============================================================
# GRADIO UI
# ============================================================

css = """
*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; -webkit-text-size-adjust: 100%; }
.gradio-container {
    padding-left: env(safe-area-inset-left);
    padding-right: env(safe-area-inset-right);
    padding-bottom: env(safe-area-inset-bottom);
}
input, textarea, select { font-size: 16px !important; }
button { min-height: 44px !important; font-size: 16px !important; cursor: pointer; -webkit-appearance: none; }
.tab-nav button {
    font-size: 14px !important;
    padding: 8px 10px !important;
    min-width: 0 !important;
    white-space: nowrap;
    flex: 1 1 0 !important;
    overflow: hidden;
    text-overflow: ellipsis;
}
.tab-nav {
    display: flex !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
}
.chatbot { height: 35vh !important; }
audio { width: 100% !important; }
@media (max-width: 768px) {
    .gradio-row { flex-direction: column !important; }
    .gradio-column { min-width: 100% !important; width: 100% !important; }
    button { font-size: 18px !important; padding: 14px !important; width: 100% !important; }
    .gradio-row button { width: auto !important; }
    .chatbot { height: 28vh !important; }
    .login-panel { padding: 16px !important; }
    .tab-nav button { font-size: 11px !important; padding: 6px 4px !important; white-space: nowrap !important; }
}
@media (prefers-color-scheme: dark) {
    .gradio-container { background-color: #1a1a1a; color: #f0f0f0; }
    .chatbot { background-color: #2a2a2a !important; }
    input, textarea { background-color: #2a2a2a !important; color: #f0f0f0 !important; }
}
@media (prefers-color-scheme: light) {
    .gradio-container { background-color: #ffffff; color: #1a1a1a; }
    .chatbot { background-color: #f9f9f9 !important; }
    input, textarea { background-color: #ffffff !important; color: #1a1a1a !important; }
}
"""

with gr.Blocks(css=css) as ui:
    gr.Markdown("Dad's Babelbot")

    gr.HTML("""
        <script>
        function checkSafariWarning() {
            const ua = navigator.userAgent;
            const isIOS = /iphone|ipad|ipod/i.test(ua);
            const isSafari = /safari/i.test(ua) && !/chrome|crios|fxios|opios|mercury/i.test(ua);
            if (isIOS && !isSafari) {
                if (document.getElementById("safari-warning")) return;
                const banner = document.createElement("div");
                banner.id = "safari-warning";
                banner.style.cssText = `
                    background-color: #ff9500; color: #000000; font-size: 15px;
                    font-weight: bold; padding: 12px 16px; text-align: center;
                    position: fixed; top: 0; left: 0; width: 100%; z-index: 9999;
                    border-bottom: 2px solid #cc7700; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                `;
                banner.innerHTML = `
                    🎤 Microphone not available in this browser on iPhone.
                    Please open this page in <u>Safari</u> to use recording features.
                    <br><span style="font-weight:normal;font-size:13px;">
                        Copy the URL and paste it into Safari.
                    </span>
                `;
                document.body.insertBefore(banner, document.body.firstChild);
                document.body.style.paddingTop = "70px";
            }
        }
        function scrollToScore() {
            const el = document.getElementById("score-box");
            if (el) { el.scrollIntoView({ behavior: "smooth", block: "start" }); }
        }
        setTimeout(checkSafariWarning, 1500);
        </script>
    """)

    # ============================================================
    # SESSION STATE
    # ============================================================
    username_state = gr.State(value=None)

    # ============================================================
    # LOGIN / REGISTER PANEL
    # ============================================================
    with gr.Column(visible=True) as login_panel:
        gr.Markdown("### 🔐 Login or Register")
        login_username = gr.Textbox(label="Username", placeholder="Enter username")
        login_password = gr.Textbox(label="Password", placeholder="Enter password", type="password")
        with gr.Row():
            login_btn    = gr.Button("Login", size="lg")
            register_btn = gr.Button("Register", size="lg")
        auth_message = gr.Textbox(label="", interactive=False, lines=1)

    # ============================================================
    # MAIN APP PANEL
    # ============================================================
    with gr.Column(visible=False) as main_panel:

        with gr.Tabs():

            # ----------------------------------------------------
            # TAB 1 — Practice
            # ----------------------------------------------------
            with gr.Tab("🗣️Learn"):
                with gr.Row():
                    with gr.Column(elem_classes="sidepanel", scale=1, min_width=200):
                        my_image = gr.Image(
                            value=image_address,
                            interactive=False,
                            height=180,
                            show_label=False)
                        voice_dropdown = gr.Dropdown(
                            choices=MALE_VOICES,
                            value="ash",
                            label="Voice",
                            interactive=True)
                    with gr.Row():
                        chatbot = gr.Chatbot(
                            label="Text Response",
                            height="40vh",
                            type="messages",
                            show_copy_button=True,
                            scale=3)
                with gr.Row():
                    audio_output = gr.Audio(
                        autoplay=False,
                        label="🔊 Speech Response")
                with gr.Column():
                    daily_btn = gr.Button("📅 Tap-to-Practice Daily Phrase", size="lg")
                    daily_audio = gr.Audio(label="Daily Phrase Audio", autoplay=False, visible=False)
                    daily_text = gr.Textbox(label="Today's Practice Phrase", lines=6, visible=False)
                    daily_visible_state = gr.State(value=False)
                    daily_phrase_state  = gr.State(value=None)
                    record_audio = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="🎤 Tap to Record Your Mandarin",
                        interactive=True)
                    submit_audio_btn = gr.Button("Submit Recorded Audio", size="lg")
                    score_output = gr.Textbox(
                        label="Professional Pronunciation Analysis",
                        lines=8,
                        elem_id="score-box")
                    message = gr.Textbox(
                        lines=2,
                        label="What would you like to translate?",
                        placeholder="Type your message...",
                        scale=4)
                    submit_btn = gr.Button("Submit", size="lg")

                gr.Examples(examples=example_msgs, inputs=message)

            # ----------------------------------------------------
            # TAB 2 — Character Pad
            # ----------------------------------------------------
            with gr.Tab("✍️Draw"):

                gr.Markdown("### ✍️ Draw a Character or Look One Up")

                with gr.Row():
                    # ------------------------------------------------
                    # LEFT COLUMN — Drawing Pad
                    # ------------------------------------------------
                    with gr.Column(scale=1):
                        gr.Markdown("**Draw a character below:**")
                        canvas_image = gr.Sketchpad(
                            label="Draw here",
                            height=400,
                            width=400,
                            brush=gr.Brush(default_size=8, colors=["#000000"], default_color="#000000"),
                        )
                        identify_btn = gr.Button("🔍 Identify Character", size="lg")

                    # ------------------------------------------------
                    # RIGHT COLUMN — Type / Paste Lookup
                    # ------------------------------------------------
                    with gr.Column(scale=1):
                        gr.Markdown("**Or type / paste a character:**")
                        char_input = gr.Textbox(
                            label="Character or phrase",
                            placeholder="e.g. 你好 or 爱",
                            lines=1,
                            max_lines=1)
                        lookup_btn = gr.Button("🔎 Look Up Character", size="lg")

                # ------------------------------------------------
                # Shared output area
                # ------------------------------------------------
                with gr.Row():
                    char_audio_output = gr.Audio(
                        autoplay=False,
                        label="🔊 Character Pronunciation")

                char_result = gr.Textbox(
                    label="Character Analysis",
                    lines=14,
                    interactive=False)

            # ----------------------------------------------------
            # TAB 3 — Progress Tracking
            # ----------------------------------------------------
            with gr.Tab("📈"):
                refresh_btn = gr.Button("🔄 Refresh Progress", size="lg")
                streak_display = gr.Textbox(
                    label="🔥 Current Streak",
                    value="Press Refresh to load",
                    interactive=False,
                    lines=1)
                summary_display = gr.Textbox(
                    label="Stats & Personal Bests",
                    value="",
                    interactive=False,
                    lines=12)
                trend_chart = gr.LinePlot(
                    value=None,
                    x="session",
                    y="score",
                    color="category",
                    title="Score Trends (last 20 sessions)",
                    x_title="Session",
                    y_title="Score (0–100)",
                    y_lim=[0, 100],
                    width=600,
                    height=350,
                    label="Score Trends")

    # ============================================================
    # EVENT HANDLER FUNCTIONS
    # ============================================================

    def handle_login(username, password):
        """Verify credentials and unlock the main app on success."""
        success, result = verify_login(username, password)
        if success:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                result,
                f"Welcome back, {result}!"
            )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            result
        )

    def handle_register(username, password):
        """Create a new account. User must then log in separately."""
        success, msg = register_user(username, password)
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            msg
        )

    def refresh_progress(username):
        """Load latest progress data for the logged-in user."""
        if not username:
            return "Please log in to view progress.", "", empty_trend_df()
        summary_text, trend_df, streak = build_progress_display(username)
        streak_text = f"🔥 {streak} day streak!" if streak > 0 else "No streak yet — practice today!"
        return streak_text, summary_text, trend_df

    # ============================================================
    # EVENT WIRING
    # ============================================================

    login_btn.click(
        handle_login,
        inputs=[login_username, login_password],
        outputs=[login_panel, main_panel, username_state, auth_message]
    )

    register_btn.click(
        handle_register,
        inputs=[login_username, login_password],
        outputs=[login_panel, main_panel, username_state, auth_message]
    )

    message.submit(
        put_message_in_chatbot,
        inputs=[message, chatbot],
        outputs=[message, chatbot]
    ).then(
        chat,
        inputs=[chatbot, voice_dropdown],
        outputs=[chatbot, audio_output]
    )

    submit_btn.click(
        put_message_in_chatbot,
        inputs=[message, chatbot],
        outputs=[message, chatbot]
    ).then(
        chat,
        inputs=[chatbot, voice_dropdown],
        outputs=[chatbot, audio_output]
    )

    daily_btn.click(
        toggle_daily_phrase,
        inputs=[daily_visible_state],
        outputs=[daily_audio, daily_text, daily_visible_state, daily_phrase_state]
    ).then(
        generate_daily_phrase,
        inputs=[daily_visible_state, voice_dropdown],
        outputs=[daily_audio, daily_text, daily_phrase_state]
    )

    submit_audio_btn.click(
        score_audio_pipeline,
        inputs=[record_audio, voice_dropdown, daily_phrase_state, username_state],
        outputs=[audio_output, score_output]
    ).then(
        None, None, None,
        js="scrollToScore()"
    )
    # drawing pad 
    identify_btn.click(
        identify_drawn_character,
        inputs=[canvas_image],
        outputs=[char_audio_output, char_result]
    )

    lookup_btn.click(
        lookup_character,
        inputs=[char_input, voice_dropdown],
        outputs=[char_audio_output, char_result]
    )

    char_input.submit(
        lookup_character,
        inputs=[char_input, voice_dropdown],
        outputs=[char_audio_output, char_result]
    )
    # progress tab refresh
    refresh_btn.click(
        refresh_progress,
        inputs=[username_state],
        outputs=[streak_display, summary_display, trend_chart]
    )

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    ui.launch()     # HuggingFace Space deployment remove inBrowser option
