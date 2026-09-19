import json

from ai.forecast import CrowdForecaster


def test_forecast_warms_up_then_projects_a_rising_trend():
    forecaster = CrowdForecaster(
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
