import json
from tempfile import TemporaryDirectory
from pathlib import Path

from ai.density import DensityAnalyzer


def _detection_at(metric_x: float, metric_y: float) -> dict:
    pixel_x = metric_x * 100
    pixel_y = metric_y * 100
    return {
        "box": [pixel_x - 1, pixel_y - 10, pixel_x + 1, pixel_y],
        "track_id": None,
        "confidence": 1.0,
    }


def test_sliding_window_detects_cluster_across_grid_boundaries():
    with TemporaryDirectory() as directory:
        analyzer = DensityAnalyzer(
            str(Path(directory) / "roi.json"), enable_local_peak=True
        )
        detections = []
        for x, y in ((0.9, 0.9), (1.1, 0.9), (0.9, 1.1), (1.1, 1.1)):
            detections.extend(_detection_at(x, y) for _ in range(3))

        result = analyzer.analyze(detections, frame_width=301, frame_height=301)

        counts = {cell["id"]: cell["count"] for cell in result["grid"]}
        assert counts[1] == 3
        assert counts[2] == 3
        assert counts[4] == 3
        assert counts[5] == 3
        assert result["density_people_per_m2"] == 12 / 9
        assert result["density_level"] == "Relaxed"
        assert result["local_peak"]["count"] == 12
        assert result["local_peak"]["density_people_per_m2"] == 12
        assert result["risk_level"] == "Danger"
        json.dumps(result)
