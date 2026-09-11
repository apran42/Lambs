import builtins
import importlib
import sys


def test_detector_contract_import_does_not_load_ai_runtimes():
    blocked = {"torch", "ultralytics", "tensorrt"}
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in blocked:
            raise AssertionError(f"Unexpected AI runtime import: {name}")
        return original_import(name, *args, **kwargs)

    try:
        builtins.__import__ = guarded_import
        sys.modules.pop("ai.detector", None)
        module = importlib.import_module("ai.detector")

        detector = module.create_detector("unavailable")
        assert detector.available is False
        assert detector.track_objects(object()) == []
        assert detector.detect_batch([object(), object()]) == [[], []]
        assert detector.health()["backend"] == "unavailable"
    finally:
        builtins.__import__ = original_import
