"""Return type utilities for Generations API.

This module provides utilities for specifying return types 
in Generation calls, particularly for common formats like JSON.
"""

from typing import Union

# JSON type aliases
JsonValue = Union[None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"]]
JsonDict = dict[str, JsonValue]
JsonList = list[JsonValue]
JsonType = Union[JsonValue, JsonDict, JsonList]

# JSON convenience constant
json = object()  # Sentinel value for JSON mode