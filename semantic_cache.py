import os
import redis
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from google.generativeai import types
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

#connecting to redis 

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False
)

#function to egenrate embeddings 

import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

def semantic_similarity(texts):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 
    
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/text-embedding-004", 
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result['embedding'])
    
    embeddings_matrix = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings_matrix)
    return similarity_matrix

texts = [
    "What is the meaning of life?",
    "What is the purpose of existence?",
    "How do I bake a cake?"
]

result = semantic_similarity(texts)
print(result)


# class semanticCache:
#     def __init__(self):
