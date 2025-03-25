"""Type-parameterized response class for LLM generations.

This module provides a `Response[T]` class that wraps LLM responses with
type information, allowing for properly typed access to the response content.
"""

from __future__ import annotations

from typing import Any, Generic, List, Optional, TypeVar, Union

from ..core.base import BaseCallResponse, BaseTool
from ..core.costs import CostMetadata

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
        cost (Optional[CostMetadata]): Cost information for the API call.
    """
    
    def __init__(
        self,
        content: T = None,  # type: ignore
        model: str = "",
        id: str = "",
        tools: Optional[List[BaseTool]] = None,
        cost: Optional[CostMetadata] = None,
        response: Optional[BaseCallResponse] = None,
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
        return cls(
            content=content,
            model=call_response.model,
            id=call_response.id if hasattr(call_response, "id") else "",
            tools=call_response.tools if hasattr(call_response, "tools") else None,
            cost=call_response.cost if hasattr(call_response, "cost") else None,
            response=call_response,
        )
    
    @property
    def input_tokens(self) -> Optional[int]:
        """Number of input tokens used in the request."""
        if self._response and hasattr(self._response, "input_tokens"):
            return self._response.input_tokens
        return None
    
    @property
    def output_tokens(self) -> Optional[int]:
        """Number of output tokens generated in the response."""
        if self._response and hasattr(self._response, "output_tokens"):
            return self._response.output_tokens
        return None
    
    @property
    def tool(self) -> Optional[BaseTool]:
        """The first tool used in the response, if any."""
        return self.tools[0] if self.tools else None
    
    @property
    def response(self) -> BaseCallResponse:
        """The underlying provider-specific response object."""
        return self._response
    
    def __str__(self) -> str:
        """String representation is the content."""
        if self.content is None:
            return ""
        return str(self.content)
    
    def __repr__(self) -> str:
        """Detailed representation of the Response object."""
        return f"Response(content={self.content}, model='{self.model}', id='{self.id}')"