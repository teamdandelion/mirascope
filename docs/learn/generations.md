---
search:
  boost: 3
---

# Generations

Mirascope Generations provide an elegant way to structure LLM code around a clear separation of concerns:
- The Prompt specifies the messages that are passed to the LLM, in a way that's provider and model agnostic.
- The Generation abstracts over the LLM call as a computational unit, providing ways to specify the return type and the execution pattern (async, streaming, etc).
- The Model context manager contains all of the model-specific configuration, ranging from the choice of model itself, to the client, model specific tuning and call parameters, etc. 

The result is an expressive, flexible, and type safe interface for writing provider-agnostic LLM code.

Here's what it looks like in practice:

```python
from mirascope import llm
from pydantic import BaseModel


# Define your prompt logic
@llm.generation()
def recommend_book(genre: str) -> str:
    return f"Recommend a {genre} book with a brief summary"


# Specify model infrastructure with a context manager
with llm.model("openai:gpt-4o-mini"):
    # Execute the generation
    response: llm.Response[str] = recommend_book(genre="fantasy")
    print(response.content) 

    # Or stream the response
    stream: llm.Stream[str] = recommend_book.stream(genre="fantasy")
    for chunk in stream:
        print(chunk.content, end="", flush=True) 
    
    # Or get structured output
    class Book(BaseModel):
        title: str
        author: str
        summary: str
    
    book_response: llm.Response[Book] = recommend_book.returns(Book)(genre="mystery")
    book: Book = book_response.content
    print(f"{book.title} by {book.author}")
```

## Generation Decorator

The `@llm.generation()` decorator defines a Generation by wrapping a prompt function. The prompt function can be as simple or complex as needed:

```python
@llm.generation()
def simple_prompt(query: str) -> str:
    return f"Answer: {query}"

@llm.generation()
def complex_prompt(query: str, context: str = None) -> str:
    if context:
        return f"""
        Use the following context to answer the question:
        
        Context: {context}
        
        Question: {query}
        """
    else:
        return f"Answer this question: {query}"
```

The `@llm.generation()` can take an argument specifying a return type, as in:

```python
from PIL import Image

@llm.generation(returns=Image)
def generate_sunset(n_moons: int) -> str:
    return f"Generate an image of a sunset, on a planet with {n_moons} moons"

with llm.model("openai:dall-e-3"):
    image_response: llm.Response[Image] = generate_sunset(n_moons=3)
    image: Image = image_response.content
    image.save("sunset.png")

class Book(BaseModel):
    title: str
    author: str
    year: int

@llm.generation(returns=Book)
def recommend_book(genre: str) -> str:
    return f"Recommend a {genre} book, providing the title, author, and year."

with llm.model("anthropic:claude-3-sonnet"):
    book_response: llm.Response[Book] = recommend_book(genre="science fiction")
    book: Book = book_response.content
    print(f"{book.title} ({book.year}) by {book.author}")

@llm.generation(returns=llm.returns.json)
def recommend_books_json(genre: str) -> str:
    return f"Recommend three {genre} books"

with llm.model("openai:gpt-4o"):
    # Returns a JSON array of books
    json_response: llm.Response[llm.returns.JsonType] = recommend_books_json(genre="fantasy")
    books: list = json_response.content
    for book in books:
        print(f"{book['title']} by {book['author']}")
```

## Model Context Manager

The `llm.model()` context manager provides the model to power your generations:

```python
# Basic usage with provider:model format
with llm.model("anthropic:claude-3-7-sonnet-latest"):
    response: llm.Response[str] = simple_prompt("What is quantum computing?")

# With custom client and model parameters
from openai import OpenAI
client = OpenAI(api_key="your-key")

with llm.model("openai:gpt-4o", client=client, temperature=0.7):
    response: llm.Response[str] = simple_prompt("What is quantum computing?")
```

A generation function must be called within a model context. This separation keeps infrastructure choices distinct from your prompt logic.

## Execution Methods

Every generation provides multiple execution methods:

### Synchronous

```python
# Basic call - returns a Response[str]
response: llm.Response[str] = recommend_book(genre="fantasy")
text: str = response.content
print(text)

# Streaming - returns a Stream[str]
stream: llm.Stream[str] = recommend_book.stream(genre="fantasy")
for chunk in stream:
    chunk_text: str = chunk.content
    print(chunk_text, end="", flush=True)
```

### Asynchronous

```python
import asyncio

# Async call - returns an awaitable Response[str]
response: llm.Response[str] = await recommend_book.acall(genre="fantasy")
text: str = response.content
print(text)

# Async streaming - returns an awaitable Stream[str]
stream: llm.Stream[str] = await recommend_book.astream(genre="fantasy")
async for chunk in stream:
    chunk_text: str = chunk.content
    print(chunk_text, end="", flush=True)
```

Note that if the prompt function is async, then only the asynchronous `acall` and `astream` methods will be functional on the Generation. 

```python
@llm.generation()
async def async_recommend_book(genre: str) -> str:
    # Async prompt function
    await asyncio.sleep(0.1)  # Simulate async work
    return f"Recommend a {genre} book"

# This won't work (will raise an error):
# response = async_recommend_book(genre="fantasy")

# Must use acall or astream:
response: llm.Response[str] = await async_recommend_book.acall(genre="fantasy")
```

## Advanced Configuration

### Tool Support

```python
from mirascope.tools import SearchTool

# Enable tool use
response: llm.Response[str] = recommend_book.with_tools([SearchTool()])(genre="obscure fantasy")

# Check tool usage
if response.tools:
    print(f"Tool used: {response.tools[0].name}")
    print(f"Tool input: {response.tools[0].input}")
```

### Call Parameters

```python
# Set provider-specific parameters
creative_response: llm.Response[str] = recommend_book.with_call_params(
    temperature=0.9,
    top_p=0.95
)(genre="fantasy")
```

## Error Handling

```python
try:
    with llm.model("openai:gpt-4o"):
        response: llm.Response[str] = recommend_book(genre="fantasy")
except Exception as e:
    print(f"Error: {e}")
    
    # Fall back to a different model
    with llm.model("anthropic:claude-3-sonnet"):
        response: llm.Response[str] = recommend_book(genre="fantasy")
```

## Advanced Usage Patterns

### Composition

```python
@llm.generation()
def summarize(text: str) -> str:
    return f"Summarize: {text}"

@llm.generation()
def translate_to_french(text: str) -> str:
    return f"Translate to French: {text}"

with llm.model("openai:gpt-4o"):
    summary_response: llm.Response[str] = summarize("Long article about quantum physics...")
    summary: str = summary_response.content
    
    french_response: llm.Response[str] = translate_to_french(summary)
    french_summary: str = french_response.content
    print(french_summary)
```

### Nested Model Contexts

```python
with llm.model("anthropic:claude-3-opus"):
    # Use Claude for the primary task
    response1: llm.Response[str] = complex_prompt(
        "Explain quantum entanglement", 
        context="Detailed physics textbook excerpt..."
    )
    explanation: str = response1.content
    
    # Switch to GPT for translation
    with llm.model("openai:gpt-4o"):
        response2: llm.Response[str] = translate_to_french(explanation)
        french_explanation: str = response2.content
```

### Testing with Mock Models

```python
# For unit tests
with llm.model("mock", responses=["The Hobbit by J.R.R. Tolkien"]):
    response: llm.Response[str] = recommend_book(genre="fantasy")
    assert "Hobbit" in response.content
```

## Next Steps

The Generations API provides a clean, flexible interface for working with LLMs that emphasizes type safety and separation of concerns. It's designed to grow with your application from simple prototypes to complex production systems.
