import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
my_key = os.environ.get("GEMINI_API_KEY")
print(my_key)

genai.configure(api_key=my_key)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config=generation_config)
