import json

from ai.forecast import CrowdForecaster


def test_forecast_warms_up_then_projects_a_rising_trend():
    forecaster = CrowdForecaster(
        horizon_seconds=300,
        history_seconds=600,
        min_trend_span_seconds=60,
        bucket_seconds=5,
    )

    first = forecaster.update(4, 4.0, 9.0, 2.0, 5.0, measured_at=0)
    assert first["method"] == "persistence-warmup"
    assert first["predicted_roi_count"] == 4
    assert first["ready"] is False

    result = first
    for second in range(5, 70, 5):
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
    assert result["horizon_seconds"] == 300
    json.dumps(result)
