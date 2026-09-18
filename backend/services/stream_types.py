from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrameSnapshot:
    frame_id: int
    captured_at: str
    frame: Any


@dataclass(frozen=True)
class StreamPacket:
    frame_id: int
    image_bytes: bytes
    metadata: dict
