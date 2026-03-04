# ============================================================
# auth_module.py
# Handles user registration, login, and password verification
# for DadBot multi-user support.
#
# Users are stored in users.json as:
#   { "username": "<hashed_password>" }
#
# Passwords are NEVER stored in plain text.
# bcrypt handles the hashing and verification.
# ============================================================

import json
import os
import bcrypt

# ------------------------------------------------------------
# Storage — local JSON file for user credentials
# Same directory as dadbot.py
# ------------------------------------------------------------
USERS_FILE = "users.json"


# ------------------------------------------------------------
# 1. Load all users from disk
# Returns dict of { username: hashed_password }
# ------------------------------------------------------------
def load_users() -> dict:
    """Load user credentials from the JSON file.
    Returns empty dict if file doesn't exist or is unreadable."""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


# ------------------------------------------------------------
# 2. Save users dict back to disk
# ------------------------------------------------------------
def save_users(users: dict):
    """Write the full users dict to the JSON file."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


# ------------------------------------------------------------
# 3. Register a new user
# Returns (success: bool, message: str)
# ------------------------------------------------------------
def register_user(username: str, password: str) -> tuple:
    """
    Register a new user with a hashed password.
    Rejects empty fields, duplicate usernames.
    Returns (True, success message) or (False, error message).
    """
    # Validate inputs
    if not username or not password:
        return False, "Username and password cannot be empty."

    if len(username) < 3:
        return False, "Username must be at least 3 characters."

    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = load_users()

    # Check for duplicate username (case-insensitive)
    if username.lower() in [u.lower() for u in users]:
        return False, "Username already exists. Please choose another."

    # Hash the password with bcrypt before storing
    # bcrypt.hashpw requires bytes, encode converts string → bytes
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    # Store as string (decode bytes → str) so JSON can serialize it
    users[username] = hashed.decode("utf-8")
    save_users(users)

    return True, f"Account created for {username}. You can now log in."


# ------------------------------------------------------------
# 4. Verify login credentials
# Returns (success: bool, message: str)
# ------------------------------------------------------------
def verify_login(username: str, password: str) -> tuple:
    """
    Check username and password against stored credentials.
    Returns (True, success message) or (False, error message).
    """
    if not username or not password:
        return False, "Please enter both username and password."

    users = load_users()

    # Find the user (case-insensitive match)
    matched_username = None
    for u in users:
        if u.lower() == username.lower():
            matched_username = u
            break

    if not matched_username:
        return False, "Username not found."

    # Retrieve stored hash and verify against entered password
    stored_hash = users[matched_username].encode("utf-8")
    password_matches = bcrypt.checkpw(password.encode("utf-8"), stored_hash)

    if not password_matches:
        return False, "Incorrect password."

    # Return the correctly-cased username for use as the session identifier
    return True, matched_username