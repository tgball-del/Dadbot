# 🤖 Dadbot — Mandarin Chinese Language Tutor

A personal AI-powered Mandarin language learning application built for my daughter's university studies. Dadbot combines conversational AI, real-time speech evaluation, and progress tracking into a fully deployed, multi-user application.

---

## 🌟 Features

### 🗣️ Conversational AI Tutor
- Powered by OpenAI GPT with a custom system prompt that enforces mandatory pinyin and per-character tone breakdowns for every Mandarin response — no exceptions
- Automatic pinyin enforcement module detects Chinese characters in any reply using Unicode regex and injects a structured pronunciation guide if the model omitted it
- Response caching layer reduces latency for repeated queries

### 📅 Daily Practice Phrases
- Fresh Mandarin phrases generated dynamically by the LLM on every session — topics rotate across greetings, food, travel, family, weather, shopping, emotions, and daily life
- Each phrase includes full pinyin, per-character tone breakdown, and pronunciation tips
- Two-step async UI: panel opens instantly with a placeholder while the LLM generates in the background

### 🎤 Speech Recording & Scoring
- Records Mandarin speech directly in the browser via microphone
- Whisper-based transcription with context priming — passes the target phrase as a prompt hint to prevent homophone misidentification
- Professional 5-category pronunciation scoring: Tone Accuracy, Pronunciation Clarity, Fluency, Grammar, Overall Score
- Coaching feedback delivered as text and spoken aloud via OpenAI TTS in a selectable voice

### 📈 Progress Tracking
- Per-user session history stored in individual JSON files
- Streak computation across consecutive practice days
- Personal bests tracked across all scoring categories
- Summary statistics and a live score trend chart across the last 20 sessions

### 🔐 Secure Multi-User Authentication
- User registration and login with bcrypt password hashing
- Passwords never stored in plain text
- Case-insensitive username matching with per-user progress file isolation

---

## 🏗️ Architecture

```
app.py                    # Main Gradio UI and event wiring
auth_module.py            # User registration, login, bcrypt hashing
daily_practice_module.py  # LLM phrase generation and TTS pipeline
progress_module.py        # Session storage, streak, stats, chart data
```

**Key engineering decisions:**
- Modular design — each concern is fully isolated in its own module
- Retry wrapper with exponential backoff for production API stability
- Cross-platform timeout protection using threading (no UNIX signal dependency)
- Whisper transcription context priming for accurate Chinese character recognition
- Full mobile optimization including iPhone safe area, iOS auto-zoom prevention, and Safari microphone compatibility detection

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT (gpt-5-mini) |
| Speech-to-Text | OpenAI Whisper (gpt-4o-mini-transcribe) |
| Text-to-Speech | OpenAI TTS (gpt-4o-mini-tts) |
| UI Framework | Gradio |
| Deployment | Hugging Face Spaces |
| Auth Security | bcrypt |
| Data | JSON, Pandas |
| Language | Python |

---

## 🚀 Running Locally

**1. Clone the repository**
```bash
git clone https://huggingface.co/spaces/TGB-007/Dadbot
cd Dadbot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your OpenAI API key**
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

**4. Run the app**
```bash
python app.py
```

---

## 📱 Mobile Support
Fully optimized for iPhone Safari including safe area insets, auto-zoom prevention, touch-friendly button sizing, and a browser compatibility warning for non-Safari iOS browsers where microphone access is restricted.

---

## 👨‍💻 Author
Built by Tomas G. Ball — enterprise software sales leader, AI practitioner, and dad.
