"""Inference contracts and dependency-safe detector construction.

This module must remain importable without torch, Ultralytics, or TensorRT. Heavy
runtime implementations are imported only after a backend has been selected.
"""

from __future__ import annotations

from typing import Any, Protocol

from config import settings


Detection = dict[str, Any]


class Detector(Protocol):
    backend_name: str
    available: bool
    unavailable_reason: str | None

    def track_objects(self, frame: Any) -> list[Detection]:
        """Return normalized detections for one frame."""

    def detect_batch(self, frames: list[Any]) -> list[list[Detection]]:
        """Return one normalized detection list per input frame."""

    def health(self) -> dict[str, Any]:
        """Describe the selected inference runtime."""


class UnavailableDetector:
    """No-op detector used while the external Jetson worker is unavailable."""

    backend_name = "unavailable"
    available = False

    def __init__(self, reason: str | None = None) -> None:
        self.unavailable_reason = reason or settings.INFERENCE_UNAVAILABLE_REASON

    def track_objects(self, _frame: Any) -> list[Detection]:
        return []

    def detect_batch(self, frames: list[Any]) -> list[list[Detection]]:
        return [[] for _frame in frames]

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "available": self.available,
            "reason": self.unavailable_reason,
        }


def create_detector(backend_name: str | None = None) -> Detector:
    """Build the selected detector without importing unused AI dependencies."""

    selected = (backend_name or settings.INFERENCE_BACKEND).strip().lower()
    if selected == "ultralytics":
        from ai.ultralytics_detector import UltralyticsDetector

        return UltralyticsDetector()
    if selected in {"unavailable", "disabled", "none"}:
        return UnavailableDetector()
    if selected == "jetson":
        return UnavailableDetector(
            "Jetson TensorRT inference worker is not connected yet."
        )
    raise ValueError(
        "Unsupported INFERENCE_BACKEND. Expected ultralytics, jetson, or unavailable: "
        f"{selected}"
    )


def PersonDetector() -> Detector:
    """Backward-compatible factory for older imports."""

    return create_detector()
