"""Layer 3: chat(role=...) for the agent, the host-wide concurrency cap, served-model recording.

- `chat_service()`: a ChatService; config.json components["inference.chat"] picks openai_compat or recorded
- `gpu_slots()`: the cap per model server; every chat call already goes through it
- `process_stats()`: per-process counters per endpoint and role
- `RecordedChat`, `Cassette`: record and replay model calls, for tests

Callers name a role, never a model: `await chat("agent", messages, tools)` on the process's shared service.
"""

from carttrace.inference.factory import chat, chat_service
from carttrace.inference.recorded import Cassette, CassetteMode, RecordedChat
from carttrace.inference.service import ChatResult, ChatService, GpuSlots, ToolCall
from carttrace.inference.slots import gpu_slots
from carttrace.inference.stats import EndpointStats, InferenceStats, process_stats

PLUGINS = {
    "inference.chat": {
        "openai_compat": "carttrace.inference.openai_chat:OpenAICompatChat",
        "recorded": "carttrace.inference.recorded:recorded_chat",
    },
}

__all__ = [
    "PLUGINS",
    "Cassette",
    "CassetteMode",
    "ChatResult",
    "ChatService",
    "EndpointStats",
    "GpuSlots",
    "InferenceStats",
    "RecordedChat",
    "ToolCall",
    "chat",
    "chat_service",
    "gpu_slots",
    "process_stats",
]
