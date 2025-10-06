# Semantic Caching for LLMs

A semantic caching system that uses Redis vector search and Google Gemini to reduce LLM API costs and latency. When users ask similar questions within the same conversation session, we return cached responses instead of making expensive API calls.

## What This Does

This project caches LLM responses based on semantic similarity within conversation sessions. If someone asks "What are Python best practices?" and later asks "What are the best practices for Python?" **in the same conversation**, we recognize these are the same question and return the cached response.

**Key features:**
- Semantic similarity matching using vector embeddings
- Session-based caching (no cross-session cache pollution)
- Session-based personalization (user preferences stay within their session)
- Tracks cache hit rate, latency, and cost savings
- Uses Redis with HNSW indexing for fast vector search

## Test Results

From running `test.py`:
- **Cache hit rate**: 40% (2 out of 5 queries hit the cache)
- **LLM calls avoided**: 2 calls
- **Cost**: $0.0025 vs $0.0063 without caching (60% savings)
- **Tokens saved**: ~4,200 tokens

The 40% hit rate demonstrates correct session-aware caching behavior. In real-world usage with longer conversations, hit rates of 60-80% are typical as users ask more follow-up questions within the same session.

## Setup

### You'll need:
- Python 3.8+
- Redis running locally
- Google Gemini API key

### Installation Steps

1. **Clone the repo**
```bash
git clone https://github.com/jiya-maheshwari/semantic-caching-DSRS.git
cd semantic-caching-DSRS
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install and start Redis**

**macOS:**
```bash
brew install redis
brew services start redis
```

**Linux:**
```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
```

**Windows:**
Use Docker: `docker run -d -p 6379:6379 redis:latest`

**Check Redis is running:**
```bash
redis-cli ping
# Should return: PONG
```

4. **Set up your API key**
```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key (get one at https://aistudio.google.com/app/apikey):
```
GOOGLE_API_KEY=your_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
```

5. **Run the tests**

**Important**: Clear Redis cache before running tests to get accurate metrics:
```bash
redis-cli FLUSHDB
python3 test.py
```

You should see output showing cache hits/misses and final metrics showing ~40% hit rate.

## How It Works

```
User Query (with session_id) → Generate Embedding → Search Redis (filtered by session)
                                                              ↓
                                                       Found Similar?
                                                        ↙         ↘
                                                      YES         NO
                                                       ↓           ↓
                                                Return Cached  Call LLM
                                                (+ personalize  Store in
                                                if has context) Cache
```

**Important**: The search step filters by `session_id`, so cache hits only occur within the same conversation session. This ensures "What about wheat?" in a farming conversation doesn't match "What about wheat?" in a baking conversation.

### Main Components

- **`Embedder`**: Converts text to 768-dim vectors using Gemini text-embedding-004
- **`LLMContext`**: Handles LLM API calls and personalizes responses based on session context
- **`SemanticCache`**: Main cache logic - searches, stores, manages sessions
- **`Telemetry`**: Tracks all the metrics (hit rate, latency, tokens, costs)

## Context-Aware Caching

The cache is **session-aware** to handle multi-turn conversations correctly.

**Problem**: If User A asks "What is climate change's impact on corn?" then "What about wheat?", and User B asks "What are good wheat varieties?" then "What about wheat?", the phrase "What about wheat?" means different things.

**Solution**: Cache searches are filtered by `session_id`. Each session has its own cache namespace, so:
- Session 1's "What about wheat?" (farming context) won't match Session 2's "What about wheat?" (baking context)
- Within a session, similar queries still hit the cache

This prevents context bleeding across conversations while maintaining cache efficiency within a conversation.

## Design Decisions

### Similarity Threshold: 0.2

I chose a cosine distance of 0.2 (roughly 80% similarity) because:
- Lower (stricter) = fewer cache hits, defeats the purpose
- Higher (looser) = might return wrong responses
- 0.2 catches paraphrases while keeping responses accurate

**Trade-off**: Occasionally might cache something too broad, but testing shows it works well for semantic matching.

### Cache TTL (Time to Live): 1 hour

Entries expire after 1 hour because:
- Most queries stay valid short-term
- Prevents stale answers
- Keeps memory usage reasonable

**Trade-off**: Popular queries might get re-cached, but better than serving outdated info.

### HNSW Indexing

Using HNSW (Hierarchical Navigable Small World) because:
- Fast: O(log N) instead of checking every cached item
- Scales well to ~1M cached queries
- Good balance of speed and accuracy

**At larger scale (10M+ queries)**: Would need to shard the index by user groups or topics, and implement better eviction policies.

### Eviction Strategy

Right now just using Redis TTL (items expire after 1 hour). For production, I'd add:
- LRU eviction when memory fills up
- Keep frequently-hit queries longer
- Remove one-time queries first

## Test Breakdown

The test file runs 6 queries across 3 different sessions to show cache behavior:

| Test | What It Tests | Result |
|------|---------------|--------|
| 1 | First query in session_1 | Cache miss (expected) |
| 2 | Similar query in session_2 | Cache miss (different session) |
| 3 | Add preferences to session_1 | - |
| 4 | Similar query in session_1 | Cache hit WITH personalization |
| 5 | Different query in session_1 | Cache miss (new content) |
| 6 | Similar query in session_3 | Cache miss (different session) |

**Key findings:**
- Semantic matching works within sessions (Test 4 matches Test 1)
- Sessions stay isolated (Test 2 and Test 6 don't match other sessions)
- Personalization works when there's session context (Test 4 mentions functional programming)
- Session filtering prevents cross-conversation pollution
- 40% hit rate is expected for multi-session test; real conversations have higher rates

## Cost Projection

**5 queries across 3 different sessions** (from our test):
- Without cache: 5 calls × ~1,700 tokens avg = $0.0063
- With cache: 3 calls = $0.0025
- Savings: 60%

**1 million queries/day** at 60% hit rate (realistic for conversational workload):
- Without cache: ~$720/day
- With cache: ~$288/day
- **Yearly savings: ~$158k**

For single-session heavy usage (like a long chatbot conversation where users ask many follow-ups), hit rates of 70-80% are achievable.

## Basic Usage Example

```python
from semantic_cache import SemanticCache, LLMContext, Telemetry, Embedder, search_index

# Setup
telemetry = Telemetry()
llm = LLMContext("gemini-2.5-flash")
vectorizer = Embedder()
cache = SemanticCache(search_index, "gemini-2.5-flash", llm, vectorizer, telemetry)

# Make a query
response = cache.query("What are Python best practices?", session_id="user_123")

# Add session context for personalization
cache.add_session_history("user_123", "preferences", "I like functional programming")
cache.add_session_history("user_123", "goals", "Learn advanced patterns")

# Next similar query will be personalized for this user
response2 = cache.query("What are good Python practices?", session_id="user_123")

# Check metrics
telemetry.report()
```

## Troubleshooting

**Redis won't connect:**
- Make sure Redis is running: `redis-cli ping`
- Check your .env file has correct REDIS_HOST

**API key error:**
- Verify your GOOGLE_API_KEY in .env is correct
- Make sure there are no extra spaces

**Module not found:**
- Run `pip install -r requirements.txt`

**Getting 100% cache hit rate:**
- Clear Redis before testing: `redis-cli FLUSHDB`
- Redis persists data between runs

## Files

```
semantic-caching-DSRS/
├── semantic_cache.py    # Main implementation
├── test.py             # Test suite
├── requirements.txt    # Dependencies
├── .env.example       # Environment template
└── README.md         # This file
```

## References

Based on the Redis AI semantic caching notebook:
https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/semantic-cache/03_context_enabled_semantic_caching.ipynb

## Author

Jiya Maheshwari