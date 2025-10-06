import os
import google.generativeai as genai
from dotenv import load_dotenv
from semantic_cache import (
    Embedder, 
    LLMContext, 
    Telemetry, 
    SemanticCache,
    search_index,
    vectorizer,
    redis_client
)
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def test_semantic_cache():
    print("="*60)
    print("SEMANTIC CACHE TEST")
    print("="*60)
    

    telemetry = Telemetry()
    llm_client = LLMContext("gemini-2.5-flash")
    cache = SemanticCache(
        redis_index=search_index,
        model="gemini-2.5-flash",
        llm_client=llm_client,
        vectorizer=vectorizer,
        telemetry=telemetry
    )

    # Test 1: Cache miss (first query)
    print("\n--- Test 1: First Query (Cache Miss) ---")
    prompt1 = "What are the best practices for Python coding?"
    response1 = cache.query(prompt1, "session_1")
    print(f"Prompt: {prompt1}")
    print(f"Response: {response1[:200]}...")
    
    # Test 2: Cache hit with similar query, different session (no personalization)
    print("\n--- Test 2: Similar Query, Different Session (Should MISS - different context) ---")
    prompt2 = "What are Python coding best practices?"
    response2 = cache.query(prompt2, "session_2")
    print(f"Prompt: {prompt2}")
    print(f"Response: {response2[:200]}...")
    
    # Test 3: Add context to session_1
    print("\n--- Test 3: Adding Session Context ---")
    cache.add_session_history("session_1", "preferences", "I prefer functional programming style")
    cache.add_session_history("session_1", "preferences", "I like using list comprehensions")
    cache.add_session_history("session_1", "goals", "Learn advanced Python design patterns")
    cache.add_session_history("session_1", "history", "Previously asked about OOP vs functional programming")
    print("Added preferences, goals, and history to session_1")
    
    # Test 4: Cache hit with personalization (same session with context)
    print("\n--- Test 4: Similar Query in Session with Context (Cache Hit - Personalized) ---")
    prompt3 = "What are the best coding practices in Python?"
    response3 = cache.query(prompt3, "session_1")
    print(f"Prompt: {prompt3}")
    print(f"Response: {response3}")
    
    # Test 5: Another query in session_1 to show personalization continues
    print("\n--- Test 5: New Query in Session 1 (Cache Miss, but has context) ---")
    prompt4 = "How do I write clean functions in Python?"
    response4 = cache.query(prompt4, "session_1")
    print(f"Prompt: {prompt4}")
    print(f"Response: {response4[:200]}...")
    
    # Test 6: Similar to Test 5 from different session (should miss cache) 
    print("\n--- Test 6: Similar to Test 5, Different Session (Should MISS--different session)")
    prompt5 = "How do I write clean Python functions?"
    response5 = cache.query(prompt5, "session_3")
    print(f"Prompt: {prompt5}")
    print(f"Response: {response5[:200]}...")
    
    # Display telemetry
    print("\n" + "="*60)
    telemetry.report()
    print("="*60)
    
    # Show session data
    print("\n--- Session Data ---")
    print(f"Session 1 history: {cache.get_session_history('session_1')}")
    print(f"Session 1 conversations: {len(cache.get_session('session_1'))} exchanges")

if __name__ == "__main__":
    test_semantic_cache()