"""Newera Python SDK: speech-to-speech calls with tool calling.

    from newera import VoiceAgent
"""

from .agent import DEFAULT_URL, Call, Event, NeweraError, Reply, ToolSpec, VoiceAgent
from .schema import function_schema, type_schema

__all__ = ["VoiceAgent", "Call", "Event", "Reply", "ToolSpec", "NeweraError", "DEFAULT_URL",
           "function_schema", "type_schema"]
__version__ = "0.1.0"
