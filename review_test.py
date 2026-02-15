#!/usr/bin/env python3
# Security test file

PASSWORD = "hardcoded_secret_123"  # Security issue

def login(user, pwd):
    # SQL injection
    return db.query(f"SELECT * FROM users WHERE name='{user}' AND pass='{pwd}'")
