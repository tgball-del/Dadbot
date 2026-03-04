# ============================================================
# progress_module.py
# Handles all progress tracking for DadBot Mandarin practice.
#
# Updated: now user-aware — each user gets their own progress
# file named dadbot_progress_<username>.json so progress is
# fully separated between users.
# ============================================================

import json
import os
import datetime
import pandas as pd


# ------------------------------------------------------------
# Helper — build the progress filename for a given user
# ------------------------------------------------------------
def get_progress_file(username: str) -> str:
    """
    Returns the progress JSON filename for the given user.
    Example: username 'sarah' → 'dadbot_progress_sarah.json'
    Lowercased to stay consistent regardless of login case.
    """
    return f"dadbot_progress_{username.lower()}.json"


# ------------------------------------------------------------
# 1. Load all saved sessions for a user
# ------------------------------------------------------------
def load_sessions(username: str) -> list:
    """Load all practice sessions for the given user.
    Returns empty list if file doesn't exist or is unreadable."""
    filepath = get_progress_file(username)
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


# ------------------------------------------------------------
# 2. Save a new session record for a user
# ------------------------------------------------------------
def save_session(username: str, phrase_mandarin: str, phrase_english: str, scores: dict):
    """
    Append a new practice session to the user's progress file.

    Args:
        username:        logged-in username — determines which file to write
        phrase_mandarin: Chinese characters of the practiced phrase
        phrase_english:  English translation (or 'free practice')
        scores:          Dict with keys: tone_accuracy,
                         pronunciation_clarity, fluency,
                         grammar, overall_score
    """
    sessions = load_sessions(username)

    session = {
        "date": datetime.date.today().isoformat(),
        "time": datetime.datetime.now().strftime("%H:%M"),
        "mandarin": phrase_mandarin,
        "english":  phrase_english,
        "tone_accuracy":         scores.get("tone_accuracy", 0),
        "pronunciation_clarity": scores.get("pronunciation_clarity", 0),
        "fluency":               scores.get("fluency", 0),
        "grammar":               scores.get("grammar", 0),
        "overall_score":         scores.get("overall_score", 0),
    }

    sessions.append(session)

    filepath = get_progress_file(username)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 3. Compute streak for a user
# ------------------------------------------------------------
def compute_streak(sessions: list) -> int:
    """
    Count how many consecutive days ending today (or yesterday)
    the user has practiced at least once.
    Returns 0 if no sessions or streak is broken.
    """
    if not sessions:
        return 0

    practice_dates = sorted(
        set(datetime.date.fromisoformat(s["date"]) for s in sessions),
        reverse=True
    )

    today = datetime.date.today()
    streak = 0

    for i, d in enumerate(practice_dates):
        expected = today - datetime.timedelta(days=i)
        if d == expected:
            streak += 1
        else:
            break

    return streak


# ------------------------------------------------------------
# 4. Compute personal bests for a user
# ------------------------------------------------------------
def compute_personal_bests(sessions: list) -> dict:
    """Return the highest score ever recorded for each category."""
    if not sessions:
        return {k: 0 for k in
                ["tone_accuracy", "pronunciation_clarity",
                 "fluency", "grammar", "overall_score"]}

    return {
        "tone_accuracy":         max(s["tone_accuracy"] for s in sessions),
        "pronunciation_clarity": max(s["pronunciation_clarity"] for s in sessions),
        "fluency":               max(s["fluency"] for s in sessions),
        "grammar":               max(s["grammar"] for s in sessions),
        "overall_score":         max(s["overall_score"] for s in sessions),
    }


# ------------------------------------------------------------
# 5. Compute summary stats for a user
# ------------------------------------------------------------
def compute_summary(sessions: list) -> dict:
    """Return total session count and average scores."""
    if not sessions:
        return {
            "total_sessions": 0,
            "avg_overall":    0,
            "avg_tone":       0,
            "avg_clarity":    0,
            "avg_fluency":    0,
            "avg_grammar":    0,
        }

    n = len(sessions)
    return {
        "total_sessions": n,
        "avg_overall":    round(sum(s["overall_score"] for s in sessions) / n, 1),
        "avg_tone":       round(sum(s["tone_accuracy"] for s in sessions) / n, 1),
        "avg_clarity":    round(sum(s["pronunciation_clarity"] for s in sessions) / n, 1),
        "avg_fluency":    round(sum(s["fluency"] for s in sessions) / n, 1),
        "avg_grammar":    round(sum(s["grammar"] for s in sessions) / n, 1),
    }


# ------------------------------------------------------------
# 6. Build chart-ready trend data for a user
# ------------------------------------------------------------
def build_trend_data(sessions: list) -> list:
    """
    Returns a list of dicts for gr.LinePlot.
    Capped at last 20 sessions for readability.
    """
    if not sessions:
        return []

    recent = sessions[-20:]

    categories = {
        "Overall":  "overall_score",
        "Tone":     "tone_accuracy",
        "Clarity":  "pronunciation_clarity",
        "Fluency":  "fluency",
        "Grammar":  "grammar",
    }

    rows = []
    for session in recent:
        label = f"{session['date']} {session['time']}"
        for display_name, key in categories.items():
            rows.append({
                "session":  label,
                "score":    session[key],
                "category": display_name,
            })

    return rows
# ================================================================
# if no user is logged in
def empty_trend_df():
    """Returns an empty DataFrame with correct columns for gr.LinePlot.
    Used when no user is logged in or no sessions exist yet."""
    return pd.DataFrame(columns=["session", "score", "category"])
# ================================================================
# ------------------------------------------------------------
# 7. Master display function — called by refresh_progress()
# ------------------------------------------------------------
def build_progress_display(username: str) -> tuple:
    """
    Returns everything the Progress tab needs for a specific user:
        summary_text:  str       — formatted stats and personal bests
        trend_df:      DataFrame — for gr.LinePlot chart
        streak:        int       — current consecutive day streak
    """
    sessions     = load_sessions(username)
    streak       = compute_streak(sessions)
    bests        = compute_personal_bests(sessions)
    summary      = compute_summary(sessions)
    trend_data   = build_trend_data(sessions)

    summary_text = f"""
📊 SUMMARY STATS
Total Sessions:  {summary['total_sessions']}
Avg Overall:     {summary['avg_overall']}
Avg Tone:        {summary['avg_tone']}
Avg Clarity:     {summary['avg_clarity']}
Avg Fluency:     {summary['avg_fluency']}
Avg Grammar:     {summary['avg_grammar']}

🏆 PERSONAL BESTS
Overall:         {bests['overall_score']}
Tone Accuracy:   {bests['tone_accuracy']}
Clarity:         {bests['pronunciation_clarity']}
Fluency:         {bests['fluency']}
Grammar:         {bests['grammar']}
""".strip()

    # Convert to DataFrame here so dadbot.py doesn't need to import pandas
    if trend_data:
        trend_df = pd.DataFrame(trend_data)
    else:
        trend_df = pd.DataFrame(columns=["session", "score", "category"])

    return summary_text, trend_df, streak