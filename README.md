# Fluent LLM Prompting Library

A Python library for fluent-style LLM interactions with built-in extraction, transformation, and retry capabilities.

## Installation

```bash
pip install openai anthropic  # Backend support
pip install pydantic  # Optional
```

## Quick Start

```python
from dataclasses import dataclass
from fluent_llm import Prompt, OpenAIBackend

@dataclass
class Person:
    name: str
    age: int

backend = OpenAIBackend(api_key="your-api-key")

# Extract structured data with fluent chaining
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
```

## Run the Demo

```bash
python demo.py
```

## Important Behavior

- `sample()` makes exactly the specified number of API calls
- `extract(schema)` takes a JSON schema, a dataclass, or a pydantic class; extracts info and returns the instance of the required type.
- `map(func)` transforms the response based on a given function
- `filter(condition)` filters instances based on a condition
- `take(n)` returns the first n items from current stream

## Testing

```bash
python -m unittest discover tests
```

## License

MIT
