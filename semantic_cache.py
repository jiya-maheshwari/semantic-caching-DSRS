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






    








