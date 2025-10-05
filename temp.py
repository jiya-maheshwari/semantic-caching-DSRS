"""
Redis Connection Test Script
Tests multiple authentication methods to find what works
"""

import os
import redis
from dotenv import load_dotenv

load_dotenv()

# Load credentials
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

print("="*60)
print("REDIS CONNECTION TEST")
print("="*60)
print(f"Host: {REDIS_HOST}")
print(f"Port: {REDIS_PORT}")
print(f"Username: {REDIS_USERNAME}")
print(f"Password: {'*' * len(REDIS_PASSWORD) if REDIS_PASSWORD else 'None'}")
print("="*60)

# Test 1: With username and password
print("\n[Test 1] Connecting with username + password...")
try:
    client1 = redis.Redis(
        host=REDIS_HOST,
        port=int(REDIS_PORT),
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        decode_responses=False
    )
    client1.ping()
    print("✓ SUCCESS with username + password")
    print("Use this method in your code!")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 2: With password only (no username)
print("\n[Test 2] Connecting with password only (no username)...")
try:
    client2 = redis.Redis(
        host=REDIS_HOST,
        port=int(REDIS_PORT),
        password=REDIS_PASSWORD,
        decode_responses=False
    )
    client2.ping()
    print("✓ SUCCESS with password only")
    print("Use this method in your code!")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 3: Using connection URL with username
print("\n[Test 3] Connecting with URL (username + password)...")
try:
    redis_url = f"redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
    client3 = redis.from_url(redis_url)
    client3.ping()
    print("✓ SUCCESS with URL (username)")
    print("Use this method in your code!")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 4: Using connection URL without username
print("\n[Test 4] Connecting with URL (password only)...")
try:
    redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
    client4 = redis.from_url(redis_url)
    client4.ping()
    print("✓ SUCCESS with URL (no username)")
    print("Use this method in your code!")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 5: Try with "default" username
print("\n[Test 5] Connecting with 'default' username...")
try:
    client5 = redis.Redis(
        host=REDIS_HOST,
        port=int(REDIS_PORT),
        username="default",
        password=REDIS_PASSWORD,
        decode_responses=False
    )
    client5.ping()
    print("✓ SUCCESS with 'default' username")
    print("Use this method in your code!")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "="*60)
print("TROUBLESHOOTING TIPS:")
print("="*60)
print("1. Go to Redis Cloud dashboard")
print("2. Click on your database")
print("3. Look for 'Data Access Control' or 'Security'")
print("4. Verify username and password are correct")
print("5. Make sure the user has 'Full Access' permissions")
print("6. Check if database status is 'Active'")
print("="*60)