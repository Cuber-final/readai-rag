import os

from llama_index.llms.deepseek import DeepSeek

# you can also set DEEPSEEK_API_KEY in your environment variables
api_key = os.getenv("DEEPSEEK_API_KEY")
llm = DeepSeek(model="deepseek-chat", api_key=api_key)

# You might also want to set deepseek as your default llm
# from llama_index.core import Settings
# Settings.llm = llm

response = llm.complete("Is 9.9 or 9.11 bigger?")
print(response)
