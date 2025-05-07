"""
test connection to OpenAI API
"""
import os
from openai import OpenAI

def main():
    """ main function to connect to OpenAI API and get a completion response """
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(
    api_key=api_key
    )
    prompt = "python code to read file and compute embeddings vector for " \
    "each line using openai library and ada-02 model."
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=False,
    messages=[
      {"role": "user", "content": prompt}
    ]
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
