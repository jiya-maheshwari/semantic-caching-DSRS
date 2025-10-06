import os
import redis
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import time
import uuid
from typing import List, Dict
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False
)

redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}" if REDIS_PASSWORD else f"redis://{REDIS_HOST}:{REDIS_PORT}"

class Embedder:
    #Generates 768-dimensional embeddings using Google Gemini text-embedding-004.
    def __init__(self,model = "models/text-embedding-004"):
        self.model = model 

    def embedding(self,text)->List[float]:
        result = genai.embed_content(
        model= self.model, 
        content=text,
        task_type="retrieval_document"
        )
        return result['embedding']
    
index_config = {
    "index": {
        "name": "cesc_index",
        "prefix": "cesc",
        "storage_type": "hash"
    },
    "fields": [
        {
            "name": "content_vector",
            "type": "vector",
            "attrs": {
                "dims": 768,
                "distance_metric": "cosine",                  
                "algorithm": "hnsw" 
            }
        },
        {"name": "content", "type": "text"},
        {"name": "session_id", "type": "tag"},
        {"name": "prompt", "type": "text"},
        {"name": "model", "type": "tag"},
        {"name": "created_at", "type": "numeric"},
    ]
}

search_index = SearchIndex.from_dict(index_config)
search_index.connect(redis_url)
search_index.create(overwrite=True)

vectorizer = Embedder() 


class LLMContext:
    """Handles LLM calls to Gemini and builds personalized responses from cached content."""
    def __init__(self,model):
        self.model = genai.GenerativeModel(model)

    def llm_call_response(self,prompt)->Dict:
        start_time = time.time()
        response = self.model.generate_content(prompt)
        latency = (time.time()-start_time)*1000
        output = response.text
        input_tokens = self.model.count_tokens(prompt).total_tokens
        output_tokens = self.model.count_tokens(output).total_tokens

        return {
            "response": output,
            "latency_ms": round(latency, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": "gemini-2.5-flash"
        }
    def build_response(self,prompt,cached_response,context):
        context_parts = []
        if context.get("preferences"):
            context_parts.append("User preferences: " + ", ".join(context["preferences"]))
        if context.get("goals"):
            context_parts.append("User goals: " + ", ".join(context["goals"]))
        if context.get("history"):
            context_parts.append("User history: " + ", ".join(context["history"]))
        context_blob = "\n".join(context_parts)
        return f"""You are a personalization assistant. A cached response was previously generated for the prompt: "{prompt}".
        Here is the cached response:\"\"\"{cached_response}\"\"\"Use the user's context below to personalize and refine the response:{context_blob}.
        Respond in a way that feels tailored to this user, adjusting tone, content, or suggestions as needed. Keep your response under 3 sentences no matter what."""

    def cache_hit_response(self,prompt,cached_response,context)->Dict:
        prompt_with_context = self.build_response(prompt,cached_response,context)
        start_time = time.time()
        response = self.model.generate_content(prompt_with_context)
        latency = (time.time()-start_time)*1000
        output = response.text
        input_tokens = self.model.count_tokens(prompt_with_context).total_tokens
        output_tokens = self.model.count_tokens(output).total_tokens

        return {
            "response": output,
            "latency_ms": round(latency, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": "gemini-2.5-flash"
        }

class Telemetry:
    #Tracks cache performance metrics: hit rate, latency, token usage, and cost.
    def __init__(self):
        self.logs = []
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.latencies = []

    def log(self, cache_status, latency_ms, input_tokens, output_tokens):
        self.latencies.append(latency_ms)
        if cache_status.startswith("hit"):
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            self.llm_input_tokens += input_tokens
            self.llm_output_tokens += output_tokens
        self.logs.append({
            "cache_status": cache_status,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })

    def report(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total else 0
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        min_latency = min(self.latencies) if self.latencies else 0
        max_latency = max(self.latencies) if self.latencies else 0

        # Gemini 2.5 Flash pricing (per 1K tokens)
        input_cost = 0.000075
        output_cost = 0.0003
        total_cost = (self.llm_input_tokens / 1000) * input_cost + (self.llm_output_tokens / 1000) * output_cost

        print(f"--- Cache Telemetry ---")
        print(f"Cache Hit Rate: {hit_rate:.2f}% ({self.cache_hits}/{total})")
        print(f"LLM Calls Avoided (Cache Hits): {self.cache_hits}")
        print(f"LLM Calls Made (Cache Misses): {self.cache_misses}")
        print(f"Tokens Generated: {self.llm_input_tokens + self.llm_output_tokens}")
        print(f"Estimated LLM Cost: ${total_cost:.4f}")
        print(f"Latency (ms): avg={avg_latency:.2f}, min={min_latency:.2f}, max={max_latency:.2f}")

class SemanticCache:
    # Session-scoped semantic cache with personalization.
    # Stores LLM responses in Redis with vector embeddings, retrieves similar queries,
    # and personalizes cached responses using session context.
    def __init__(self, redis_index,model, llm_client: "LLMContext", vectorizer, telemetry:"Telemetry", cache_ttl=3600):
        self.redis_index = redis_index
        self.model = model
        self.vectorizer = vectorizer
        self.llm = llm_client
        self.telemetry = telemetry
        self.cache_ttl = cache_ttl
        self.sessions : Dict[str, List[Dict]] = {}
        self.session_history : Dict[str,Dict] = {}

    def add_session_history(self,session_id,memory_type,content):
        #Store user context (preferences, goals, history) for personalization.
        if session_id not in self.session_history:
            self.session_history[session_id] = {"preferences": [], "goals": [], "history": []}
        self.session_history[session_id][memory_type].append(content)

    def add_session(self,session_id,prompt,response):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"prompt": prompt, "response": response})

    def get_session_history(self,session_id):
        return self.session_history.get(session_id, {})

    def get_session(self,session_id):
        return self.sessions.get(session_id, [])

    def get_all_session_history(self):
        return self.session_history
    
    def get_all_sessions(self):
        return self.sessions
    
    def generate_embedding(self,text):
        return self.vectorizer.embedding(text)
    
    def search_cache(self,embedding,session_id,distance_threshold=0.2):
    # Search for semantically similar cached responses within the same session.
    # Returns cached result if cosine distance <= threshold, else None.
        return_fields = ["content", "session_id", "prompt", "model", "created_at"]
        query = VectorQuery(
            vector=embedding,
            vector_field_name="content_vector",
            return_fields=return_fields,
            num_results=1,
            return_score=True,
            filter_expression=Tag("session_id") == session_id
        )
        results = self.redis_index.query(query)

        if results:
            first = results[0]
            score = first.get("vector_distance", None)
            if score is not None and float(score) <= distance_threshold:
                return {field: first[field] for field in return_fields}
        return None
    
    def store_response(self, prompt: str, response: str, embedding: List[float], session_id: str, model: str):
        #Store LLM response in Redis with vector embedding and TTL expiration.
        vec_bytes = np.array(embedding, dtype=np.float32).tobytes()

        doc = {
            "content": response,
            "content_vector": vec_bytes,
            "session_id": session_id,
            "prompt": prompt,
            "model": "gemini-2.5-flash",
            "created_at": int(time.time())
        }

        key = f"{self.redis_index.prefix}:{uuid.uuid4()}"
        self.redis_index.load([doc], keys=[key])

        if self.cache_ttl > 0:
            redis_client = self.redis_index.client
            redis_client.expire(key, self.cache_ttl)

    def query(self, prompt: str,session_id: str):
        start_time = time.time()
        embedding = self.generate_embedding(prompt)
        cached_result = self.search_cache(embedding,session_id)

        session_context = self.get_session_history(session_id)

        if cached_result:
            cached_response = cached_result["content"]
            
            if session_context:
                result = self.llm.cache_hit_response(prompt, cached_response, session_context)
                response_text = result["response"]
                self.telemetry.log(
                cache_status="hit_personalized",
                latency_ms=result["latency_ms"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"]
                )
            else:
                cache_latency = (time.time() - start_time) * 1000
                response_text = cached_response
                self.telemetry.log(
                cache_status="hit_raw",
                latency_ms=round(cache_latency, 2),
                input_tokens=0,
                output_tokens=0
                )
        else:
            result = self.llm.llm_call_response(prompt)
            response_text = result["response"]
            self.store_response(prompt, response_text, embedding,session_id,result["model"])
            self.telemetry.log(
            cache_status="miss",
            latency_ms=result["latency_ms"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"]
            )

        if session_id:
            self.add_session(session_id, prompt, response_text)

        return response_text



















    








