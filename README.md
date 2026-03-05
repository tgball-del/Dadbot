# Dad's Babelbot

Dad's Babelbot is an interactive Mandarin Chinese learning assistant built with Gradio and OpenAI APIs. It combines AI conversation, pronunciation coaching, daily phrase practice, character recognition, and progress tracking into a single learning experience.

The app is designed to support structured learning while keeping interaction natural and engaging.

---

# 🚀 Features

## 1. AI Mandarin Chat (with Mandatory Pinyin Enforcement)

* Conversational Mandarin support
* Automatic pinyin with tone marks
* Per-character pronunciation breakdown
* Tone description for every character
* Text-to-speech playback

Every Chinese phrase returned includes:

* Chinese characters
* Full pinyin (with tone marks)
* Per-character breakdown with tone explanation

---

## 2. 🎤 Professional Pronunciation Scoring

Users can record their Mandarin speech and receive structured evaluation.

### Scoring Categories (0–100):

* Tone Accuracy
* Pronunciation Clarity
* Fluency
* Grammar
* Overall Score

Each submission includes:

* Transcript
* Structured rubric scoring
* Strengths
* Areas for improvement
* Coaching feedback
* Spoken encouragement

Daily phrase attempts are stored for progress tracking.

---

## 3. 📅 Daily Practice Phrase

* One-tap phrase generation
* Native-style pronunciation audio
* Full structured breakdown
* Designed for consistent daily improvement

Daily attempts contribute to streak tracking and trend analysis.

---

## 4. ✍️ Character Drawing Recognition

Users can draw a Chinese character using the built-in sketchpad.

The app:

* Identifies the intended character
* Provides pinyin with tone marks
* Shows stroke count
* Lists radical
* Provides example usage
* Gives memory tips

Alternatively, users can type or paste a character for lookup.

---

## 5. 📈 Progress Tracking

Each logged-in user has:

* Session history
* Score trends (last 20 sessions)
* Personal best tracking
* Streak counter

Progress data is stored per user.

---

# 🧠 Architecture Overview

## Frontend

* Gradio UI
* Multi-tab layout
* Responsive CSS
* iPhone Safari microphone compatibility notice

## Backend

* OpenAI Chat API for conversational logic
* OpenAI Vision model for character recognition
* OpenAI TTS for audio responses
* Whisper-based transcription for pronunciation scoring
* JSON-based structured scoring output

## Core Modules

* `app.py` — Main application
* `progress_module.py` — Progress storage and trend generation
* `auth_module.py` — User authentication
* `daily_practice_module.py` — Daily phrase generation

---

# 🛠️ Installation

## 1. Clone Repository

```
git clone <your-repo-url>
cd <repo-folder>
```

## 2. Install Dependencies

```
pip install -r requirements.txt
```

Required packages include:

* gradio
* openai
* python-dotenv
* pillow
* numpy
* pandas

## 3. Set Environment Variable

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

## 4. Run Locally

```
python app.py
```

---

# 🔐 Authentication

Users must register and log in.

Features unlocked after login:

* Pronunciation tracking
* Streak tracking
* Score history
* Trend visualization

---

# ⚙️ Configuration

Key configuration values in `app.py`:

* `MODEL = "gpt-4.1-mini"`
* Vision model: `gpt-4o`
* TTS model: `gpt-4o-mini-tts`
* Transcription model: `gpt-4o-mini-transcribe`

Timeout and retry logic is implemented to prevent hanging API calls.

---

# 🎯 Design Goals

* Enforce correct Mandarin pronunciation
* Provide structured tone education
* Encourage daily repetition
* Track measurable improvement
* Keep UX simple and responsive

---

# 🧩 Future Improvements

Potential enhancements:

* Hybrid local + AI character matching
* HSK level filtering
* Vocabulary tracking by category
* Exportable progress reports
* Admin dashboard

---

# 📜 License

No license required right now

---

# 👨‍💻 Author

Tomas Ball - created for my daughter
