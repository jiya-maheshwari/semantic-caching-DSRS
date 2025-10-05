import os 
import redis 
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost") 
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "Jiya")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")  

redis_client = redis.Redis(
  host=REDIS_HOST,
  port=REDIS_PORT,
  username=REDIS_USERNAME,
  password=REDIS_PASSWORD
)

redis_client.ping()

