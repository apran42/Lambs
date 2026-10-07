import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai.count_gru import (CountGRUForecaster, NumpyCountGRU, build_history,
                          select_video_horizon)


class CountGRUTests(unittest.TestCase):
    def test_rolling_forecaster_selects_video_and_live_models(self):
        with tempfile.TemporaryDirectory() as directory:
            for horizon, correction in [(30, 0.1), (60, 0.2)]:
                model_dir = Path(directory) / ("video%d_runtime" % horizon)
                model_dir.mkdir()
                (model_dir / "model.json").write_text(
                    '{"format":"lambs-count-gru-v1",'
                    '"horizon_video_seconds":%d,"history_seconds":30,'
                    '"stride_seconds":10}' % horizon, encoding="utf-8")
                np.savez(model_dir / "weights.npz", **{
                    "gru.weight_ih_l0": np.zeros((48, 2), dtype=np.float32),
                    "gru.weight_hh_l0": np.zeros((48, 16), dtype=np.float32),
                    "gru.bias_ih_l0": np.zeros(48, dtype=np.float32),
                    "gru.bias_hh_l0": np.zeros(48, dtype=np.float32),
                    "head.weight": np.zeros((1, 16), dtype=np.float32),
                    "head.bias": np.asarray([correction], dtype=np.float32),
                })
            video = CountGRUForecaster(directory)
            live = CountGRUForecaster(directory)
            for second in (0, 10, 20, 30):
                video_result = video.update(5, 1, 5, 2, 5,
                                            measured_at=second * 2,
                                            source_seconds=second,
                                            source_fps=30, capture_fps=15)
                live_result = live.update(5, 1, 5, 2, 5,
                                          measured_at=second)
            self.assertEqual(video_result["predicted_roi_count"], 6)
            self.assertEqual(video_result["model_video_horizon_seconds"], 30)
            self.assertEqual(live_result["predicted_roi_count"], 7)
            self.assertEqual(live_result["model_video_horizon_seconds"], 60)
            self.assertTrue(video_result["ready"])
            self.assertEqual(video.update(5, 1, 5, 2, 5, measured_at=65,
                                          source_seconds=1, source_fps=30,
                                          capture_fps=15)["ready"], False)

    def test_video_horizon_accounts_for_playback_speed(self):
        self.assertEqual(select_video_horizon(60, 30, 15), (30, 60.0))
        self.assertEqual(select_video_horizon(60, 30, 20), (40, 60.0))
        self.assertEqual(select_video_horizon(60, 30, 30), (60, 60.0))
        self.assertIsNone(select_video_horizon(60, 30, 10))

    def test_history_matches_training_sampling(self):
        samples = [(0, 2), (8, 3), (19, 4), (30, 5)]
        result = build_history(samples, 30, history_seconds=30,
                               stride_seconds=10)
        np.testing.assert_allclose(result, [
            [0.2, 0.0], [0.3, 0.2], [0.4, 0.1], [0.5, 0.0]
        ], atol=1e-6)

    def test_history_rejects_missing_or_unordered_samples(self):
        self.assertIsNone(build_history([(0, 1)], 30, 30, 10))
        with self.assertRaisesRegex(ValueError, "increasing"):
            build_history([(10, 1), (9, 2)], 30, 30, 10)

    def test_numpy_gru_matches_pytorch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable; NumPy runtime remains usable")
        torch.manual_seed(7)
        gru = torch.nn.GRU(input_size=2, hidden_size=16, batch_first=True)
        head = torch.nn.Linear(16, 1)
        weights = {"gru." + key: value.detach().numpy()
                   for key, value in gru.state_dict().items()}
        weights.update({"head." + key: value.detach().numpy()
                        for key, value in head.state_dict().items()})
        history = np.asarray([[0.2, 0], [0.3, 0.2], [0.4, 0.1], [0.5, 0]],
                             dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            np.savez(path, **weights)
            with torch.no_grad():
                _, hidden = gru(torch.tensor(history).unsqueeze(0))
                expected = 5.0 + head(hidden[-1])[0, 0].item() * 10.0
            actual = NumpyCountGRU(path).predict(history, 5)
        self.assertAlmostEqual(actual, expected, places=5)


if __name__ == "__main__":
    unittest.main()
