import os
import redis
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from google.generativeai import types
from sklearn.metrics.pairwise import cosine_similarity
import time
import uuid
from typing import List, Dict
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

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

## Embedder Class and object
class Embedder:
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
        {"name": "user_id", "type": "tag"},
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
    def __init__(self,model):
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def llm_call_response(self,model,prompt)->Dict:
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
            "model": model
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
        start_time = time()
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
            "model": self.model
        }


















    








