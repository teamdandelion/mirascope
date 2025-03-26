"""Context manager for LLM model environments.

This module provides a `model` context manager that sets up the execution
environment for LLM generations, including provider, model, and call parameters.
"""

from __future__ import annotations

import threading
from types import TracebackType
from typing import Any

from ..core.base import CommonCallParams

# Context stack to store current model environment
_context_stack = threading.local()


class ModelEnvironment:
    """Environment configuration for model execution.

    Attributes:
        provider (str): The LLM provider (e.g., "openai", "anthropic").
        model (str): The specific model to use.
        client (Any): Optional custom client for the provider.
        call_params (Dict[str, Any]): Additional parameters for the provider call.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        client: Any | None = None,  # noqa: ANN401
        call_params: CommonCallParams | Any = None,  # noqa: ANN401
    ) -> None:
        """Initialize a ModelEnvironment.

        Args:
            provider: The LLM provider.
            model: The specific model to use.
            client: Optional custom client for the provider.
            **call_params: Additional parameters for the provider call.
        """
        self.provider = provider
        self.model = model
        self.client = client
        self.call_params = call_params


def _init_context_stack() -> None:
    """Initialize the context stack if it doesn't exist."""
    if not hasattr(_context_stack, "stack"):
        _context_stack.stack = []


def _get_current_environment() -> ModelEnvironment | None:
    """Get the current model environment from the context stack.

    Returns:
        The current model environment, or None if no context is active.
    """
    _init_context_stack()
    return _context_stack.stack[-1] if _context_stack.stack else None


class ModelContextManager:
    """Context manager for model execution environments."""

    def __init__(
        self,
        model_spec: str,
        client: Any | None = None,  # noqa: ANN401
        call_params: CommonCallParams | Any = None,  # noqa: ANN401
    ) -> None:
        """Initialize the model context manager.

        Args:
            model_spec: The model specification in format "provider:model".
            client: Optional custom client for the provider.
            **kwargs: Additional parameters for the provider call.

        Raises:
            ValueError: If the model_spec is not in the correct format.
        """
        _init_context_stack()

        # Parse the model spec
        if ":" not in model_spec:
            raise ValueError(
                f"Invalid model specification: {model_spec}. "
                "Format should be 'provider:model', e.g. 'openai:gpt-4o'"
            )

        provider, model_name = model_spec.split(":", 1)

        # Create the environment
        self.environment = ModelEnvironment(
            provider=provider,
            model=model_name,
            client=client,
            call_params=call_params,
        )

    def __enter__(self) -> ModelEnvironment:
        """Enter the context manager.

        Returns:
            The model environment.
        """
        _context_stack.stack.append(self.environment)
        return self.environment

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager."""
        _context_stack.stack.pop()


def model(
    model_spec: str,
    client: Any | None = None,  # noqa: ANN401
    call_params: CommonCallParams | Any = None,  # noqa: ANN401
) -> ModelContextManager:
    """Context manager for configuring the model execution environment.

    This context manager sets up the environment for executing generations
    by specifying the provider, model, and other parameters.

    Args:
        model_spec: The model specification in format "provider:model".
        client: Optional custom client for the provider.
        **kwargs: Additional parameters for the provider call.

    Returns:
        A context manager for the model environment.

    Raises:
        ValueError: If the model_spec is not in the correct format.
    """
    return ModelContextManager(model_spec, client, call_params)
