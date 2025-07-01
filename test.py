import os
from dotenv import load_dotenv

dotenv_path = os.path.abspath(".env")
print("Looking for .env at:", dotenv_path)

load_dotenv(dotenv_path=dotenv_path)

openai_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY:", openai_key if openai_key else "❌ Not found")
