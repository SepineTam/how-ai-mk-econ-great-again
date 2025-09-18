import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY")
)

model_list: List[str] = ["THE MODELS YOU WANT TO USE"]

def test_api():
    """"
    If you want to test your api, you can use the free model for a test
    """
    free_model = "deepseek/deepseek-chat-v3.1:free"

    response = client.chat.completions.create(
        model=free_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the meaning of life?"},
        ],
    )
    print(response)


if __name__ == "__main__":
    print(len(model_list))

