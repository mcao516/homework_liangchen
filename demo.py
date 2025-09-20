# demo.py
import asyncio
from dataclasses import dataclass
import os

from llm_fluent.backends.openai import OpenAIBackend
from llm_fluent.prompt import Prompt


# It is recommended to set the OPENROUTER_API_KEY environment variable
# You can get your key from https://openrouter.ai/keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class Person:
    name: str
    age: int


def main():
    """Demonstrates the synchronous usage of the llm-fluent library."""
    print("--- Synchronous Demo ---")
    backend = OpenAIBackend(
        base_url=BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model="meta-llama/llama-3-8b-instruct",
    )

    corpus = """
    John is a 30-year-old software engineer.
    Mary, a 25-year-old doctor, lives in the same town.
    Peter, aged 17, is a high school student.
    """
    prompt_text = f"Find all the people's names and their ages that appear in the corpus:\n{corpus}"

    result = (
        Prompt(prompt=prompt_text, backend=backend)
        .sample()
        .extract(Person)
        .filter(lambda p: p.age > 18)
        .take(2)
    )

    for person in result:
        print(f"Name: {person.name}, Age: {person.age}")


async def amain():
    """Demonstrates the asynchronous usage of the llm-fluent library."""
    print("\n--- Asynchronous Demo ---")
    backend = OpenAIBackend(
        base_url=BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model="meta-llama/llama-3-8b-instruct",
    )

    corpus = """
    Alice is a 42-year-old data scientist.
    Bob, a 19-year-old artist, is her neighbor.
    Charlie, aged 35, is a project manager.
    """
    prompt_text = f"Find all the people's names and their ages that appear in the corpus:\n{corpus}"

    result = (
        await Prompt(prompt=prompt_text, backend=backend)
        .asample()
        .aextract(Person)
        .afilter(lambda p: p.age < 40)
        .atake(2)
    )

    for person in result:
        print(f"Name: {person.name}, Age: {person.age}")


if __name__ == "__main__":
    main()
    asyncio.run(amain())