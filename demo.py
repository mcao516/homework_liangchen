"""A demonstration of the llm-fluent library using OpenRouter."""

import asyncio
import logging
import os
from dataclasses import dataclass

from llm_fluent import OpenAIBackend
from llm_fluent.prompt import Prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Person:
    """Example dataclass for extraction."""
    name: str = ""
    age: int = 0


@dataclass
class Product:
    """Example dataclass for extraction."""
    name: str = ""
    price: float = 0.0


def demo_sync():
    """Demonstrate synchronous usage."""
    print("=== Synchronous Demo ===\n")
    
    backend = OpenAIBackend(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    
    corpus = """
    In our company, we have several employees:
    Alice is 30 years old and works in engineering.
    Bob, age 25, is in marketing.
    Charlie is our 17-year-old intern.
    David, who is 40, manages the team.
    We are a great team.
    """
    prompt = f"Extract people's names and ages from:\n{corpus}\nReturn a JSON array."
    
    result = (
        Prompt(prompt=prompt, backend=backend)
        .sample()
        .extract(Person)
        .filter(lambda p: p.age > 18)
        .map(lambda p: Person(name=p.name.upper(), age=p.age))
        .take(3)
    )
    
    print(f"Prompt: {prompt}\n")
    print("Extracted people (adults only, first 2):")
    for person in result:
        print(f"  - {person.name}: {person.age} years old")


async def demo_async():
    """Demonstrate asynchronous usage."""
    print("\n=== Asynchronous Demo ===\n")
    
    backend = OpenAIBackend(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    
    corpus = """### Fresh on the Market: A Quick Look at New Tech

From your morning coffee to your creative pursuits, a new lineup of gadgets has arrived. Here's a glimpse at what's new and what it will cost you.

*   **The "Aura" Smart Mug: $129.99**
    This mug doesn't just hold your coffee; it keeps it at your ideal temperature for hours and displays personalized images on its digital surface.

*   **"Chrono-Leap" Gaming Console: $599.99**
    Experience a new dimension of gaming. The "Chrono-Leap" console introduces "4D" gaming, incorporating subtle shifts in room temperature and scent to match your gameplay.

*   **"Bio-Loom" Self-Repairing Fabric: $45 per yard**
    A revolutionary step in sustainable fashion, this smart textile uses microscopic organisms to slowly weave itself back together when torn.

*   **"Echo-Scribe" Pen: $75.00**
    For those who love to write by hand, the "Echo-Scribe" seamlessly bridges the gap between analog and digital by instantly saving your handwritten notes to the cloud."""
    prompt = f"""Extract all product names and prices from:\n{corpus}\nReturn a JSON array of objects with two keys: "name" (str) and "price" (float)."""
    
    result = await (
        Prompt(prompt=prompt, backend=backend)
        .asample()
        .extract(Product)
        .filter(lambda p: p.price < 150)
        .take(3)
    )
    
    print(f"Prompt: {prompt}\n")
    print("Extracted products (price < $150, first 3):")
    for product in result:
        print(f"  - {product.name}: ${product.price}")


if __name__ == "__main__":
    demo_sync()
    
    asyncio.run(demo_async())
