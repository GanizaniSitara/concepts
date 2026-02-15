#!/usr/bin/env python3
"""Test file for AI review system"""

import os

# Intentional issue: hardcoded password
PASSWORD = "mysecretpassword123"

def connect_db():
    # Missing error handling
    conn = db.connect(f"postgresql://user:{PASSWORD}@localhost/mydb")
    return conn

def process_user_input(user_input):
    # Potential SQL injection
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return query
