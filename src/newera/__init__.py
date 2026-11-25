"""Newera Python SDK: speech-to-speech calls with tool calling, and the talking avatar.

    from newera import VoiceAgent, Newera
"""

from .agent import DEFAULT_URL, Call, Event, NeweraError, Reply, ToolSpec, VoiceAgent
from .avatar import (AudioDelta, Availability, AvatarBusy, AvatarCall, AvatarError, AvatarEvent, AvatarReady,
                     AvatarReply, AvatarUnavailable, AuthenticationError, CallTimeLimit, ClientSecret, ErrorEvent,
                     InsufficientQuota, Interrupted, InvalidRequest, Line, LineDone, LineError, LineStarted, NotEnabled, OtherEvent, RateLimited,
                     ReplyText, ReplyTextDelta, ResponseDone, SessionSummary, SpeechStarted, SpeechStopped,
                     Timeline, Unavailable, UserTranscript, VideoChunk, WrongMode)
from .client import Newera
from .h264 import DecodedFrame, H264Decoder
from .player import LipSyncPlayer
from .schema import function_schema, type_schema


def talk(**kwargs) -> int:
    """An interactive call with the avatar (microphone, speaker, window). See `newera avatar --help`.
    Needs `pip install "newera-sdk[avatar]"`."""
    from .talk import talk as _talk
    return _talk(**kwargs)


__all__ = ["VoiceAgent", "Call", "Event", "Reply", "ToolSpec", "NeweraError", "DEFAULT_URL",
           "function_schema", "type_schema",
           "Newera", "talk", "AvatarCall", "AvatarReply", "Availability", "ClientSecret", "Timeline",
           "AvatarEvent", "AvatarReady", "VideoChunk", "Interrupted", "Unavailable", "AudioDelta", "UserTranscript",
           "ReplyTextDelta", "ReplyText", "SpeechStarted", "SpeechStopped", "ResponseDone", "SessionSummary",
           "ErrorEvent", "OtherEvent",
           "AvatarError", "InvalidRequest", "AuthenticationError", "InsufficientQuota", "NotEnabled",
           "CallTimeLimit", "RateLimited", "AvatarBusy", "AvatarUnavailable",
           "WrongMode", "LineError", "Line", "LineStarted", "LineDone", "LipSyncPlayer", "H264Decoder", "DecodedFrame"]
__version__ = "0.5.0"
