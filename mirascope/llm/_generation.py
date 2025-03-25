"""The `generation` decorator and `Generation` class.

The `generation` decorator creates a provider-agnostic interface for LLM interactions
by wrapping a prompt function and returning a `Generation` object. The `Generation`
offers a fluent API for configuring execution and output formats.

Key components:
- `generation()` decorator: Wraps a prompt function
- `Generation` class: Fluent interface for LLM execution
- `llm.model()` context manager: Sets execution environment

Design principles:
1. Separate prompt logic from execution configuration
2. Method chaining for clear composition of behaviors
3. Type safety through generics
4. Explicit execution methods (call/stream/acall/astream)

Implementation notes:
- Each configuration method returns a new Generation instance
- The Generation class is generic over both input and output types
- The Generation uses the context from llm.model() for execution
- Async prompt functions only support async execution methods

Internal architecture:
- The prompt function dictates input parameters
- The `returns` parameter and configuration methods determine output type
- Execution methods respect the function's synchronicity
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Generic,
    Iterable,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from ..core.base import BaseTool
from ._response import Response
from .stream import Stream

# Type variables for generics
P = TypeVar("P", contravariant=True)  # Parameters
T = TypeVar("T")  # Output type (parameterized)
R = TypeVar("R")  # Return type of wrapped function


class Generation(Generic[P, T]):
    """Fluent interface for LLM execution.

    The Generation class wraps a prompt function and provides methods for
    configuring execution and output formats. Each configuration method returns
    a new Generation instance with the updated configuration.

    Attributes:
        _prompt_fn: The wrapped prompt function
        _returns: Optional return type for structured output
    """

    def __init__(
        self,
        prompt_fn: Callable[P, R],
        returns: Optional[Type[T]] = None,
        tools: Optional[list[BaseTool]] = None,
        json_mode: bool = False,
    ) -> None:
        """Initialize a Generation object.

        Args:
            prompt_fn: The prompt function to wrap
            returns: Optional return type for structured output
            tools: Optional list of tools for the LLM to use
            json_mode: Whether to use JSON mode
        """
        self._prompt_fn = prompt_fn
        self._returns = returns
        self._tools = tools

    # Configuration methods - each returns a new Generation with updated config

    def returns(self, model_type: Type[T]) -> "Generation[P, T]":
        """Configure the generation to return a structured response model.

        Args:
            model_type: The type to structure the response as

        Returns:
            A new Generation configured to return the specified type
        """
        return Generation(
            self._prompt_fn,
            returns=model_type,
            tools=self._tools,
        )

    def with_tools(self, tools: list[BaseTool]) -> "Generation[P, T]":
        """Configure the generation to use the specified tools.

        Args:
            tools: The tools to make available to the LLM

        Returns:
            A new Generation configured to use the specified tools
        """
        return Generation(
            self._prompt_fn,
            returns=self._returns,
            tools=tools,
        )

    # Execution methods - these perform the actual LLM call

    def stream(self, *args: P.args, **kwargs: P.kwargs) -> Stream[T]:
        """Execute the generation with streaming enabled.

        Args:
            *args: Positional arguments for the prompt function
            **kwargs: Keyword arguments for the prompt function

        Returns:
            A Stream object containing the streaming response
        """
        # Stub implementation - actual impl would interact with provider
        return Stream()  # type: ignore

    async def acall(self, *args: P.args, **kwargs: P.kwargs) -> Response[T]:
        """Execute the generation asynchronously.

        Args:
            *args: Positional arguments for the prompt function
            **kwargs: Keyword arguments for the prompt function

        Returns:
            A Response object containing the response
        """
        # Stub implementation - actual impl would interact with provider
        return Response()  # type: ignore

    async def astream(self, *args: P.args, **kwargs: P.kwargs) -> Stream[T]:
        """Execute the generation with streaming enabled and asynchronously.

        Args:
            *args: Positional arguments for the prompt function
            **kwargs: Keyword arguments for the prompt function

        Returns:
            A Stream object containing the streaming response
        """
        # Stub implementation - actual impl would interact with provider
        return Stream()  # type: ignore

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Response[T]:
        """Execute the generation synchronously (alias for call).

        Args:
            *args: Positional arguments for the prompt function
            **kwargs: Keyword arguments for the prompt function

        Returns:
            A Response object containing the response
        """
        return self.call(*args, **kwargs)

    def call(self, *args: P.args, **kwargs: P.kwargs) -> Response[T]:
        """Execute the generation synchronously.

        Args:
            *args: Positional arguments for the prompt function
            **kwargs: Keyword arguments for the prompt function

        Returns:
            A Response object containing the response
        """
        # Stub implementation - actual impl would interact with provider
        return Response()  # type: ignore


def generation(
    returns: Optional[Type[T]] = None,
    **kwargs: Any,
) -> Callable[[Callable[P, R]], Generation[P, T]]:
    """Decorator that transforms a function into a Generation object.

    This decorator wraps a prompt function and returns a Generation object
    that provides a fluent interface for configuring execution and output formats.

    Args:
        returns: Optional return type for structured output
        **kwargs: Additional configuration options for the Generation

    Returns:
        A decorator function that creates a Generation from a prompt function
    """

    def decorator(fn: Callable[P, R]) -> Generation[P, T]:
        """The actual decorator function.

        Args:
            fn: The prompt function to wrap

        Returns:
            A Generation object wrapping the prompt function
        """
        return Generation(fn, returns=returns)

    return decorator
