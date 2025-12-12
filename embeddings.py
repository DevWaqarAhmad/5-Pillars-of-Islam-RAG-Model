import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import embeddings

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))