# LLM Fluent

A fluent-style LLM prompting and response transformation library for Python.

This library provides a clean, chainable interface to interact with Large Language Models, process their responses, and extract structured data.

## Features

- **Fluent Interface**: A simple, intuitive API: `.sample().extract().filter().take()`.
- **Multi-Backend Support**: Designed to work with various LLM providers (e.g., OpenAI, Anthropic).
- **Structured Data Extraction**: Automatically parse LLM responses into `dataclasses` or `pydantic` models.
- **Lazy Evaluation**: Operations are performed lazily, making it efficient for sampling and processing large streams of data.
- **Async Support**: Fully asynchronous version available for high-performance applications.

## Installation

Install the library using pip:
```
pip install "openai>=1.0.0" pydantic
```