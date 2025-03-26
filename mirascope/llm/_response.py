"""Type-parameterized response class for LLM generations.

This module provides a `Response[T]` class that wraps LLM responses with
type information, allowing for properly typed access to the response content.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from ..core.base import BaseCallResponse, BaseTool

T = TypeVar("T")  # Response content type parameter


class Response(Generic[T]):
    """Typed response from an LLM generation.
    
    This class wraps an LLM response and provides typed access to its content.
    The type parameter T specifies the type of the content property.
    
    Attributes:
        content (T): The main content of the response, typed according to T.
        model (str): The name of the model that generated the response.
        id (str): A unique identifier for the response.
        tools (List[BaseTool]): Any tools used by the model in generating the response.
        cost (Optional[float]): Cost information for the API call.
    """
    
    def __init__(
        self,
        content: T | None = None,
        model: str = "",
        id: str = "",
        tools: list[BaseTool] | None = None,
        cost: float | None = None,
        response: BaseCallResponse | None = None,
    ) -> None:
        """Initialize a Response object.
        
        Args:
            content: The typed content of the response.
            model: The name of the model that generated the response.
            id: A unique identifier for the response.
            tools: Any tools used by the model in generating the response.
            cost: Cost information for the API call.
            response: The underlying provider-specific response object.
        """
        self.content = content
        self.model = model
        self.id = id
        self.tools = tools or []
        self.cost = cost
        self._response = response
    
    @classmethod
    def from_call_response(cls, call_response: BaseCallResponse, content: T) -> Response[T]:
        """Create a typed Response from a provider-specific CallResponse.
        
        Args:
            call_response: The provider-specific response object.
            content: The typed content for the response.
            
        Returns:
            A new Response object with the typed content.
        """
        # Extract values with safe defaults
        model = call_response.model if hasattr(call_response, "model") and call_response.model is not None else ""
        id = call_response.id if hasattr(call_response, "id") and call_response.id is not None else ""
        tools = call_response.tools if hasattr(call_response, "tools") and call_response.tools is not None else []
        cost = call_response.cost if hasattr(call_response, "cost") and call_response.cost is not None else None
        
        return cls(
            content=content,
            model=model,
            id=id,
            tools=tools,
            cost=cost,
            response=call_response,
        )
    
    @property
    def input_tokens(self) -> int | None:
        """Number of input tokens used in the request."""
        if self._response and hasattr(self._response, "input_tokens"):
            tokens = self._response.input_tokens
            return int(tokens) if tokens is not None else None
        return None
    
    @property
    def output_tokens(self) -> int | None:
        """Number of output tokens generated in the response."""
        if self._response and hasattr(self._response, "output_tokens"):
            tokens = self._response.output_tokens
            return int(tokens) if tokens is not None else None
        return None
    
    @property
    def tool(self) -> BaseTool | None:
        """The first tool used in the response, if any."""
        return self.tools[0] if self.tools else None
    
    @property
    def response(self) -> BaseCallResponse:
        """The underlying provider-specific response object."""
        if self._response is None:
            raise ValueError("No underlying response object is available")
        return self._response
    
    def __str__(self) -> str:
        """String representation is the content."""
        if self.content is None:
            return ""
        return str(self.content)
    
    def __repr__(self) -> str:
        """Detailed representation of the Response object."""
        return f"Response(content={self.content}, model='{self.model}', id='{self.id}')"