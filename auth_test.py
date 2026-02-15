#!/usr/bin/env python3
# Test file with intentional security issues

API_KEY = "sk_live_secret_1234567890"  # Hardcoded secret

def authenticate(username, password):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE name='{username}' AND pass='{password}'"
    return db.execute(query)
