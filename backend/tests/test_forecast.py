import json
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime

from ai.forecast import CrowdForecaster


def test_reviewed_demo_uses_matching_video_timeline_and_falls_back_safely():
    with TemporaryDirectory() as directory:
        reference = Path(directory) / "reference.json"
        reference.write_text(
            json.dumps({
                "horizon_seconds": 60,
                "videos": {
                    "11": {
                        "duration_seconds": 180,
                        "samples": [[0, 1], [30, 3], [60, 6], [70, 7], [90, 9]],
                    }
                },
            }),
            encoding="utf-8",
        )
        forecaster = CrowdForecaster(mode="reviewed-demo", reviewed_demo_path=str(reference))
        result = forecaster.update(
            2, 0.5, 10, 2, 5, measured_at=10,
            source_video_id="11", video_time_seconds=10,
        )
        assert result["predicted_roi_count"] == 7
        assert result["method"] == "reviewed-video-demo"
        assert result["confidence"] is None
        assert result["reference_video_time_seconds"] == 70
        slowed_playback = forecaster.update(
            2, 0.5, 10, 2, 5, measured_at=15,
            source_video_id="11", video_time_seconds=30,
            video_seconds_per_wall_second=2 / 3,
        )
        assert slowed_playback["reference_video_time_seconds"] == 70
        assert slowed_playback["predicted_roi_count"] == 7
        wrapped = forecaster.update(
            2, 0.5, 10, 2, 5, measured_at=20,
            source_video_id="11", video_time_seconds=150,
        )
        assert wrapped["predicted_roi_count"] == 3
        fallback = forecaster.update(
            2, 0.5, 10, 2, 5, measured_at=30,
            source_video_id="99", video_time_seconds=10,
        )
        assert fallback["predicted_roi_count"] == 2
        assert fallback["method"] == "persistence-baseline"
        mismatched_video = forecaster.update(
            2, 0.5, 10, 2, 5, measured_at=40,
            source_video_id="11", video_time_seconds=10,
            source_duration_seconds=120,
        )
        assert mismatched_video["method"] == "persistence-baseline"


def test_default_one_minute_baseline_keeps_latest_count_without_claiming_confidence():
    forecaster = CrowdForecaster()
    first = forecaster.update(4, 4.0, 9.0, 2.0, 5.0, measured_at=0)
    later = forecaster.update(7, 7.0, 9.0, 2.0, 5.0, measured_at=60)

    assert first["predicted_roi_count"] == 4
    assert later["predicted_roi_count"] == 7
    assert later["method"] == "persistence-baseline"
    assert later["ready"] is True
    assert later["horizon_seconds"] == 60
    assert later["assumption"] == "current-count-unchanged"
    assert later["confidence"] is None
    assert later["confidence_label"] == "unvalidated"
    assert (
        datetime.fromisoformat(later["forecast_for"])
        - datetime.fromisoformat(later["generated_at"])
    ).total_seconds() == 60
    json.dumps(later)


def test_forecast_warms_up_then_projects_a_rising_trend():
    forecaster = CrowdForecaster(
        mode="trend",
        horizon_seconds=60,
        history_seconds=120,
        min_trend_span_seconds=30,
        bucket_seconds=5,
    )

    first = forecaster.update(4, 4.0, 9.0, 2.0, 5.0, measured_at=0)
    assert first["method"] == "persistence-warmup"
    assert first["predicted_roi_count"] == 4
    assert first["ready"] is False

    result = first
    for second in range(5, 40, 5):
        count = 4 + second // 20
        result = forecaster.update(
            count,
            float(count),
            9.0,
            2.0,
            5.0,
            measured_at=second,
        )

    assert result["method"] == "damped-linear-trend"
    assert result["ready"] is True
    assert result["predicted_roi_count"] > 4
    assert result["horizon_seconds"] == 60
    json.dumps(result)


def test_forecast_keeps_only_the_rolling_history_window():
    forecaster = CrowdForecaster(
        horizon_seconds=60,
        history_seconds=120,
        min_trend_span_seconds=30,
        bucket_seconds=5,
    )

    result = None
    for second in range(0, 181, 5):
        result = forecaster.update(
            count=second // 30,
            local_peak_density=float(second // 30),
            zone_area_m2=9.0,
            relaxed_max=2.0,
            danger_min=5.0,
            measured_at=second,
        )

    assert result is not None
    assert result["ready"] is True
    assert result["history_span_seconds"] <= 120
    assert result["sample_count"] == 25


def test_unknown_forecast_mode_is_rejected():
    try:
        CrowdForecaster(mode="unknown")
    except ValueError:
        return
    raise AssertionError("Unknown forecast modes must be rejected")
