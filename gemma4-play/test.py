import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(override=True)

# Initialize Ollama client
ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

# Model options (uncomment the one you want to use)
# model_name = "llama3.2"
model_name = "gemma4:e4b"
# model_name = "gemma4:26b"
# model_name = "gemma4:31b"

# Define the messages for the chat completion
messages = [
    {"role": "user", "content": "Please come up with a challenging, nuanced question that I can ask an LLM to evaluate its intelligence. Answer only with the question, no explanation."}
]

# Make the chat completion call
response = ollama.chat.completions.create(model=model_name, messages=messages)
answer = response.choices[0].message.content

# Print the answer
print(answer)

# to run: source .venv/bin/activate && python3 gemma4-play/test.py