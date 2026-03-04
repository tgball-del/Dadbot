
#imports
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import re  # needed for Chinese character detection
import pandas as pd  # needed to convert trend data for gr.LinePlot
# progress tracking, authorization, and daily practice modules
from progress_module import save_session, build_progress_display, empty_trend_df
from auth_module import register_user, verify_login
from daily_practice_module import daily_phrase_pipeline

# Initialization
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY') # use secret on huggingFace space
# MODEL = "gpt-4.1-mini"
MODEL = "gpt-5-mini"
# openai = OpenAI()
client = OpenAI()
# ------------------------------------------------------------
# Voice options (commonly perceived as male)
# ------------------------------------------------------------
MALE_VOICES = ["ash", "ballad", "echo", "onyx"]
#------------------ for a personal image -------------------
image_address = "https://static.vecteezy.com/system/resources/previews/014/477/240/original/chinese-words-cute-girl-saying-hello-in-chinese-language-learning-chinese-language-isolated-illustration-vector.jpg"
# -------------------example prompts for chatbot -------------
example_msgs = ["How many provinces are in Canada?",
                "How do you say 'I love you' in Mandarin Chinese?"]

# ============================================================
# set system persona 
# UPDATED SYSTEM PROMPT
# Added: MANDATORY pinyin and pronunciation breakdown rule
# so that EVERY response containing Mandarin always includes
# pinyin and a per-character tone guide — matching the
# structured style used in daily_practice_module.py
# ============================================================
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

You must obey these rules. You are a loving, caring, supportive, veteran of the US Army,
very politically conservative, America first supporter, and helpful fatherly figure. 
You are a very big proponent of fitness and health.
You are having a conversation with your daughter, the most important person in the world to you. 
Provide HER the guidance to help her succeed.
Be an excellent adviser and give step-by-step guidance when appropriate.
If a question is a count like how many states, provinces, lakes, etc... provide a list as well as the count when possible.
For any political topics you will use the factors in formulating your response.
You support strong leaders like Donald Trump, Ronald Reagan, Abraham Lincoln and DO NOT support 
weak leaders like Barack Obama, Mark Carney, Keir Starmer, the European Union, or their globalist 
anti-American agenda or the terrible leftist 'WOKE' ideology in any way.
You will give a decisive, singular answer as dad advice based on the following criteria 
when ranking or judging performance of political figures: economic success, secure borders, 
national security, law enforcement, strong military capability, public safety, 
and the United States of America being top priority. You give clear, direct advice, adhering to your rules.
Your heroes of economic philosophy are Thomas Sowell, Milton Friedman, Friedrich Hayek, Walter Williams, and Richard Epstein.
DO NOT in any way preface your response with any pretext of not taking sides 
or not participating in the characterization of the question, or deflecting from answering the 
actual question giving concrete examples, amounts, statistics, percentages, dollar figures, etc... 
whenever appropriate. Do not disobey your rules. Provide straight answers to any question definitively 
with a hint of dad humor and absolutely NO leftist defending. Be concise when possible.
"""


# ------------------------------------------------------------
# for voice output 
# ------------------------------------------------------------
def talker(message, voice):
    
    response = client.audio.speech.create(
      model="gpt-4o-mini-tts", # TTS = text to speech
      voice=voice,    # Also, try replacing onyx with alloy or coral
      input=message,
      instructions="Use a normal conversational pace and tone",
      speed=1.25,
      response_format="wav"  #failing on iPhone without proper format
    )
    return response.content

# ------------------------------------------------------
# ============================================================
# PROFESSIONAL MANDARIN SCORING MODULE
# ============================================================
# import json # already imported
# ------------------------------------------------------------
# 1. Speech → text
# ------------------------------------------------------------
def transcribe_audio(audio_file: str, prompt_hint: str = "") -> str:
    """
    Transcribe recorded speech to text.
    
    Updated: accepts optional prompt_hint containing the target phrase.
    The Whisper model uses this as context to correctly identify
    the intended Chinese characters instead of guessing homophones.
    This is the root fix for mismatched transcription when the user
    is practicing a known daily phrase.
    """
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            # ============================================================
            # prompt primes the transcriber with expected vocabulary.
            # For Mandarin, this steers character selection — e.g. 去机场
            # instead of 舊之長 when the sounds are ambiguous.
            # ============================================================
            prompt=prompt_hint if prompt_hint else None,
            language="zh"   # ← explicitly tell Whisper this is Mandarin Chinese
        ).text
    return transcript

# ============================================================
# PINYIN ENFORCEMENT MODULE
# Added to ensure every assistant reply containing Mandarin
# characters is automatically supplemented with a structured
# pinyin + pronunciation breakdown, mirroring the format used
# in daily_practice_module.py
# ============================================================

# import re  # needed for Chinese character detection # imported with other imports at beginning

def contains_chinese(text: str) -> bool:
    """
    Returns True if the text contains any Chinese/Mandarin characters.
    Uses Unicode range U+4E00–U+9FFF which covers CJK Unified Ideographs.
    This is our trigger to enforce a pinyin breakdown.
    """
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def extract_chinese_phrases(text: str) -> list:
    """
    Extracts all sequences of Chinese characters from the reply text.
    Returns a deduplicated list of unique Chinese phrases found.
    Example: "你好 means hello, 谢谢 means thank you" → ["你好", "谢谢"]
    """
    # Find all contiguous runs of Chinese characters (1 or more)
    phrases = re.findall(r'[\u4e00-\u9fff]+', text)
    # Deduplicate while preserving order
    seen = set()
    unique_phrases = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique_phrases.append(p)
    return unique_phrases


def generate_pinyin_breakdown(phrases: list) -> str:
    """
    Calls the OpenAI model to generate a structured pinyin + tone breakdown
    for each Chinese phrase found in the assistant's reply.
    
    Returns a formatted string block ready to append to the reply.
    This mirrors the per-character breakdown style in daily_practice_module.py
    but is dynamically generated for any phrase the model produces.
    """
    if not phrases:
        return ""

    # Build a prompt asking for structured breakdown of each unique phrase
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
    Master enforcement function called on every assistant reply.
    
    Steps:
    1. Check if the reply contains any Chinese characters
    2. If yes, extract all unique Chinese phrases
    3. Generate a structured pinyin breakdown for those phrases
    4. Append the breakdown block to the reply if it adds NEW information
       (i.e., the model didn't already include a full breakdown)
    
    This is the safety net that guarantees pinyin is always shown,
    even if the model partially forgot to include it.
    """
    if not contains_chinese(reply):
        return reply  # No Chinese characters — nothing to do

    # Extract all unique Chinese phrases from the reply
    phrases = extract_chinese_phrases(reply)

    # Heuristic: if the reply already contains pinyin tone marks (ā á ǎ à etc.)
    # AND has bullet points, assume the model did a good job and skip injection
    # to avoid duplicating content. Otherwise, always inject.
    has_tone_marks = bool(re.search(r'[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]', reply))
    has_breakdown_bullets = '•' in reply or '·' in reply

    if has_tone_marks and has_breakdown_bullets:
        # Model already included a structured breakdown — trust it
        return reply

    # Generate and append the breakdown
    breakdown = generate_pinyin_breakdown(phrases)
    if breakdown:
        # Append with a clear visual separator so it's easy to read
        reply += f"\n\n---\n📖 **Pronunciation Guide**\n{breakdown}"

    return reply
# ------------------------------------------------------------
# 2. Structured Mandarin evaluation
# ------------------------------------------------------------
def evaluate_mandarin(transcript: str) -> dict:
    """
    Perform professional Mandarin pronunciation evaluation.
    Returns structured rubric JSON.
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


# ------------------------------------------------------------
# 3. Human-readable feedback formatter
# ------------------------------------------------------------
def format_feedback(transcript: str, scores: dict) -> str:
    """Create clean coaching text for UI display."""

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


# ------------------------------------------------------------
# 4. Full scoring pipeline (Gradio entry point)
# ------------------------------------------------------------
def score_audio_pipeline(audio_file, voice, target_phrase=None, username=None):
    """
    End-to-end pipeline:
    audio → transcript → rubric scores → formatted feedback → TTS

    target_phrase: dict from daily_phrase_state (may be None for free practice)
    username:      logged-in username passed from gr.State for progress tracking
    """

    if audio_file is None:
        return None, "No audio recorded."

    # ============================================================
    # Build transcription hint from target phrase if available.
    # Giving Whisper the expected characters as a prompt prevents
    # it from hallucinating wrong homophones.
    # ============================================================
    transcription_hint = ""
    if target_phrase:
        transcription_hint = (
            f"The speaker is practicing this Mandarin phrase: "
            f"{target_phrase['mandarin']} "
            f"({' '.join(target_phrase['pinyin'])})"
        )

    # Step 1: Transcribe — with context hint if available
    transcript = transcribe_audio(audio_file, prompt_hint=transcription_hint)

    # Step 2: Evaluate — score only what the user actually said
    scores = evaluate_mandarin(transcript)

    # Step 3: Format the score block
    feedback_text = format_feedback(transcript, scores)

    # ============================================================
    # PRONUNCIATION REFERENCE BLOCK
    # If practicing a daily phrase: show the TARGET phrase breakdown.
    # If free practice: generate breakdown from the transcript.
    # ============================================================
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

    # ============================================================
    # PROGRESS TRACKING: Save session only if a user is logged in.
    # Skips saving for unauthenticated sessions.
    # ============================================================
    if username:
        phrase_mandarin = target_phrase["mandarin"] if target_phrase else transcript
        phrase_english  = target_phrase["english"]  if target_phrase else "(free practice)"
        save_session(username, phrase_mandarin, phrase_english, scores)

    # Step 4: Speak the coaching feedback aloud
    spoken_feedback = talker(scores["coach_feedback"], voice)

    return spoken_feedback, feedback_text
#-------------------------------------------------------
# handle chat context and tools
# ------------------------------------------------------
def put_message_in_chatbot(message, history):
        return "", history + [{"role":"user", "content":message}] # move user input into chatbot area
    
def chat(history, voice): # input chat history and voice selection from dropdown
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_prompt}] + history
    response = client.chat.completions.create(model=MODEL, messages=messages) # add parameter: tools=tools (if any tools are defined)
    cities = []
    image = None
  
    while response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        responses, cities = handle_tool_calls_and_return_cities(message)
        messages.append(message)
        messages.extend(responses)
        response = client.chat.completions.create(model=MODEL, messages=messages) # removed tools=tools parameter with no tools defined
    # updated reply to use pinyin enforcement
    # reply = response.choices[0].message.content
    # history += [{"role":"assistant", "content":reply}]

    # Get the raw reply from the model
    reply = response.choices[0].message.content
    # ============================================================
    # PINYIN ENFORCEMENT: Run every reply through the enforcement
    # function. If Chinese characters are present and a full
    # breakdown wasn't already included, one will be appended.
    # This mirrors the daily_practice_module.py breakdown format.
    # ============================================================
    reply = enforce_pinyin_in_reply(reply)
    history += [{"role":"assistant", "content":reply}]
    speech = talker(reply, voice) # generate audio

    # if cities: # if the user provided any cities use 1st city (cities[0]) to generate an image
    #     image = artist(cities[0])
    
    return history, speech  # outputs listed - removed image
#--------------------------recorded audio submission ---------
# Handle recorded audio → transcription → scoring → TTS reply
# ------------------------------------------------------------
def score_audio(audio_file, voice):

    if audio_file is None:
        return None, "No audio provided."

    # 1. Transcribe speech → text
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            language="zh"   # ← explicitly declare Mandarin Chinese
        ).text

    # 2. Ask model to score pronunciation / language quality
    scoring_prompt = f"""
You are grading a Mandarin learner.

Transcript of student's speech:
{transcript}

Provide:
1. Pronunciation score (0–100)
2. Fluency score (0–100)
3. Brief improvement advice
Be concise and encouraging.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": scoring_prompt}]
    )

    score_text = response.choices[0].message.content
    # ============================================================
    # PINYIN ENFORCEMENT: Append pinyin breakdown for any Chinese
    # characters found in the student's transcript.
    # Keeps scoring output consistent with the rest of the app.
    # ============================================================
    pinyin_block = generate_pinyin_breakdown(
        extract_chinese_phrases(transcript)
    )
    if pinyin_block:
        score_text += f"\n\n---\n📖 PRONUNCIATION REFERENCE\n{pinyin_block}"
    # ============================================================
    # 3. Convert score feedback → spoken audio
    speech = talker(score_text, voice)

    return speech, score_text
# ------------------------------------------------------------
# allows show/hide of daily phrase components
#-------------------------------------------------------------
def toggle_daily_phrase(state: bool, voice: str):
    """
    Toggles the visibility of daily practice components.
    Generates a fresh LLM phrase only when showing (not on hide).
    Returns phrase_info as 4th output so daily_phrase_state
    is populated for use by score_audio_pipeline().
    """
    new_state = not state  # flip visibility
    audio, text = (None, None)
    phrase_info = None  # default when hiding

    if new_state:
        # ============================================================
        # Generate phrase separately so we can store phrase_info
        # in daily_phrase_state for the scoring pipeline to reference.
        # ============================================================
        from daily_practice_module import get_daily_phrase_from_llm, build_daily_phrase_text, speak_daily_phrase
        phrase_info = get_daily_phrase_from_llm(client, MODEL)
        text = build_daily_phrase_text(phrase_info)
        audio = speak_daily_phrase(phrase_info, talker, voice)

    return (
        gr.update(value=audio, visible=new_state),
        gr.update(value=text, visible=new_state),
        new_state,    # update daily_visible_state
        phrase_info   # update daily_phrase_state for scorer
    )

# ------------------------------------------ Tools------------
# not being used right now - stub for future tools

def handle_tool_calls_and_return_cities(message):
    responses = []
    cities = []
    # for tool_call in message.tool_calls:
    #     if tool_call.function.name == "get_ticket_price":
        #     arguments = json.loads(tool_call.function.arguments)
        #     city = arguments.get('destination_city')
        #     cities.append(city)
        #     price_details = get_ticket_price(city)
        #     responses.append({
        #         "role": "tool",
        #         "content": price_details,
        #         "tool_call_id": tool_call.id
        #     })
    return responses, cities # return to... chat, inputs=chatbot, outputs=[chatbot, audio_output, image_output]
# ------------------------------------------------------------
# Tools function json description exmaple - required for each tool being used
# ------------------------------------------------------------
# price_function = {
#     "name": "get_ticket_price",
#     "description": "Get the price of a return ticket to the destination city.",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "destination_city": {
#                 "type": "string",
#                 "description": "The city that the customer wants to travel to",
#             },
#         },
#         "required": ["destination_city"],
#         "additionalProperties": False
#     }
# }
# tools = [{"type": "function", "function": price_function}]

# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------
# trying to improve appearance on iOS/mobile device
# ------------------ Mobile CSS ------------------
css = """
/* ============================================================
   BASE STYLES
   Smooth scrolling and box sizing for all elements
   ============================================================ */
*, *::before, *::after {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
    /* Prevent iOS font size adjustment on orientation change */
    -webkit-text-size-adjust: 100%;
}

/* ============================================================
   SAFE AREA — iPhone notch and home bar
   Ensures content is never hidden behind iPhone hardware
   ============================================================ */
.gradio-container {
    padding-left:   env(safe-area-inset-left);
    padding-right:  env(safe-area-inset-right);
    padding-bottom: env(safe-area-inset-bottom);
}

/* ============================================================
   INPUTS — prevent iOS auto-zoom
   iPhone zooms in when input font-size is under 16px.
   Setting 16px on all inputs prevents this behavior.
   ============================================================ */
input, textarea, select {
    font-size: 16px !important;
}

/* ============================================================
   BUTTONS — large touch targets for iPhone
   Apple recommends minimum 44x44pt tap targets.
   ============================================================ */
button {
    min-height: 44px !important;
    font-size: 16px !important;
    cursor: pointer;
    /* Prevent iOS button styling override */
    -webkit-appearance: none;
}

/* ============================================================
   TABS — prevent cramping on narrow screens
   ============================================================ */
.tab-nav button {
    font-size: 14px !important;
    padding: 8px 10px !important;
    min-width: 0 !important;
    white-space: nowrap;
}

/* ============================================================
   CHATBOT — responsive height
   ============================================================ */
.chatbot {
    height: 35vh !important;
}

/* ============================================================
   AUDIO PLAYER — full width on mobile for easier tap
   ============================================================ */
audio {
    width: 100% !important;
}

/* ============================================================
   MOBILE OVERRIDES (max-width: 768px)
   iPhone-specific layout adjustments
   ============================================================ */
@media (max-width: 768px) {

    /* Stack columns vertically on small screens */
    .gradio-row {
        flex-direction: column !important;
    }

    /* Full width columns on mobile */
    .gradio-column {
        min-width: 100% !important;
        width: 100% !important;
    }

    /* Larger buttons for easier tapping */
    button {
        font-size: 18px !important;
        padding: 14px !important;
        width: 100% !important;
    }

    /* Chatbot shorter on mobile to leave room for keyboard */
    .chatbot {
        height: 28vh !important;
    }

    /* Login panel padding on small screens */
    .login-panel {
        padding: 16px !important;
    }

    /* Tabs slightly smaller text on narrow screens */
    .tab-nav button {
        font-size: 13px !important;
        padding: 6px 8px !important;
    }
}

/* ============================================================
   LIGHT / DARK MODE — follow system setting
   Uses CSS variables that Gradio respects.
   ============================================================ */
@media (prefers-color-scheme: dark) {
    .gradio-container {
        background-color: #1a1a1a;
        color: #f0f0f0;
    }
    .chatbot {
        background-color: #2a2a2a !important;
    }
    input, textarea {
        background-color: #2a2a2a !important;
        color: #f0f0f0 !important;
    }
}

@media (prefers-color-scheme: light) {
    .gradio-container {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    .chatbot {
        background-color: #f9f9f9 !important;
    }
    input, textarea {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
}
"""

with gr.Blocks(css=css) as ui:
    gr.Markdown("Dad's Babelbot")

    # ============================================================
    # username_state holds the logged-in username for the session.
    # None means no user is logged in.
    # Passed to score_audio_pipeline() and refresh_progress()
    # so progress is always saved/loaded per user.
    # ============================================================
    username_state = gr.State(value=None)

    # ============================================================
    # LOGIN / REGISTER PANEL
    # Shown on startup, hidden after successful login.
    # Register creates a new account.
    # Login verifies credentials and stores username in state.
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
    # Hidden on startup, shown after successful login.
    # ============================================================
    with gr.Column(visible=False) as main_panel:

        with gr.Tabs():

            # ----------------------------------------------------
            # TAB 1 — Practice (all existing UI unchanged)
            # ----------------------------------------------------
            with gr.Tab("🗣️ Practice"):
                with gr.Row():
                    with gr.Column(elem_classes="sidepanel", scale=1, min_width=200):
                        my_image = gr.Image(value=image_address,
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
                            label="Chat Text Response",
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
                    message = gr.Textbox(lines=2,
                        label="Ask me anything?",
                        placeholder="Type your message...",
                        scale=4)
                    submit_btn = gr.Button("Submit", size="lg")
                    gr.HTML("""
                    <script>
                    /* ============================================================
                    scrollToScore — scrolls to the scoring output box after
                    audio is submitted. Unchanged from original.
                    ============================================================ */
                    function scrollToScore() {
                        const el = document.getElementById("score-box");
                        if (el) {
                            el.scrollIntoView({ behavior: "smooth", block: "start" });
                        }
                    }
                    </script>

                    <script>
                    /* ============================================================
                        Safari detection — runs on page load.
                        Shows a warning banner ONLY when on iOS and NOT using Safari.
                        isSafari checks for Safari user agent while excluding Chrome
                        and other browsers that spoof the Safari UA string on iOS.
                        ============================================================ */
                    document.addEventListener("DOMContentLoaded", function() {
                        const ua = navigator.userAgent;
                        const isIOS = /iphone|ipad|ipod/i.test(ua);
                        const isSafari = /safari/i.test(ua) && !/chrome|crios|fxios|opios|mercury/i.test(ua);

                        if (isIOS && !isSafari) {
                            // Create the warning banner
                            const banner = document.createElement("div");
                            banner.id = "safari-warning";
                            banner.innerHTML = `
                                <div style="
                                    background-color: #ff9500;
                                    color: #000000;
                                    font-size: 15px;
                                    font-weight: bold;
                                    padding: 12px 16px;
                                    text-align: center;
                                    position: sticky;
                                    top: 0;
                                    z-index: 9999;
                                    border-bottom: 2px solid #cc7700;
                                    ">
                                        🎤 Microphone not available in this browser on iPhone.
                                        Please open this page in <u>Safari</u> to use recording features.
                                        <br>
                                        <span style="font-weight: normal; font-size: 13px;">
                                            Copy the URL and paste it into Safari.
                                        </span>
                                    </div>
                            `;
                            // Insert at the very top of the page body
                            document.body.insertBefore(banner, document.body.firstChild);
                        }
                    });
                    </script>
                    """)

                gr.Examples(examples=example_msgs, inputs=message)

            # ----------------------------------------------------
            # TAB 2 — Progress Tracking
            # ----------------------------------------------------
            with gr.Tab("📈 Progress"):
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
    # LOGIN / REGISTER EVENT HANDLERS
    # On success: hide login panel, show main app, store username.
    # On failure: show error message, keep login panel visible.
    # ============================================================
    def handle_login(username, password):
        """Verify credentials and unlock the main app on success."""
        success, result = verify_login(username, password)
        # print(f"DEBUG login result: success={success} result='{result}'")  # ← ADD THIS LINE for debug
        if success:
            return (
                gr.update(visible=False),  # hide login panel
                gr.update(visible=True),   # show main app
                result,                    # store username in state
                f"Welcome back, {result}!"
            )
        else:
            return (
                gr.update(visible=True),   # keep login panel visible
                gr.update(visible=False),  # keep main app hidden
                None,                      # no username stored
                result                     # show error message
            )

    def handle_register(username, password):
        """Create a new account and show result in auth message."""
        success, msg = register_user(username, password)
        if success:
            return (
                gr.update(visible=True),   # keep login visible — user must now log in
                gr.update(visible=False),  # main app still hidden
                None,                      # no username stored yet
                msg                        # show success message
            )
        else:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                None,
                msg                        # show error message
            )

    # ============================================================
    # PROGRESS TAB REFRESH FUNCTION
    # Loads latest data from the user's progress file and
    # populates all three display components in the Progress tab.
    # ============================================================
    def refresh_progress(username):
        """Load progress data for the logged-in user."""
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

    message.submit(put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
        chat, inputs=[chatbot, voice_dropdown], outputs=[chatbot, audio_output]
    )
    submit_btn.click(put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
        chat, inputs=[chatbot, voice_dropdown], outputs=[chatbot, audio_output])

    daily_btn.click(
        toggle_daily_phrase,
        inputs=[daily_visible_state, voice_dropdown],
        outputs=[daily_audio, daily_text, daily_visible_state, daily_phrase_state])

    submit_audio_btn.click(
        score_audio_pipeline,
        inputs=[record_audio, voice_dropdown, daily_phrase_state, username_state],
        outputs=[audio_output, score_output]
    ).then(
        None, None, None,
        js="scrollToScore()"
    )

    refresh_btn.click(
        refresh_progress,
        inputs=[username_state],
        outputs=[streak_display, summary_display, trend_chart]
    )

# ------------------------------------------------------------
# Run 
# ------------------------------------------------------------
if __name__ == "__main__":
    # ui.launch(inbrowser=True)
    ui.launch() # used for deployment to hugging face space