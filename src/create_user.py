#!/usr/bin/env python3
"""
CLI to create or reset a TennisPredict user.
Password is always prompted interactively — never passed as a CLI argument.

Usage:
    python src/create_user.py --email user@example.com
    python src/create_user.py --email user@example.com --reset
"""
import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.webapp.db import get_connection, init_db
from src.webapp.auth import hash_password


def main():
    parser = argparse.ArgumentParser(description="Manage TennisPredict users")
    parser.add_argument("--email", required=True, help="User email address")
    parser.add_argument("--reset", action="store_true", help="Reset password for existing user")
    args = parser.parse_args()

    email = args.email.strip().lower()

    password = getpass.getpass(f"Password for {email}: ")
    if len(password) < 8:
        print("Password must be at least 8 characters.")
        sys.exit(1)

    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords do not match.")
        sys.exit(1)

    conn = get_connection()
    init_db(conn)

    existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()

    if args.reset:
        if not existing:
            print(f"No user found with email: {email}")
            sys.exit(1)
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?",
            (hash_password(password), email),
        )
        conn.commit()
        print(f"Password updated for {email}")
    else:
        if existing:
            print(f"User already exists: {email}. Use --reset to change password.")
            sys.exit(1)
        conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, hash_password(password)),
        )
        conn.commit()
        print(f"User created: {email}")

    conn.close()


if __name__ == "__main__":
    main()
