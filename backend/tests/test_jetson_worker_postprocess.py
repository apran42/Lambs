import unittest

import numpy as np

from jetson_worker.postprocess import decode_yolo_output, prepare_input


class JetsonWorkerPostprocessTests(unittest.TestCase):
    def test_prepare_input_letterboxes_to_static_nchw(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, transform = prepare_input(image, (1, 3, 640, 640))

        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(transform["scale"], 1.0)
        self.assertEqual(transform["pad_x"], 0)
        self.assertEqual(transform["pad_y"], 80)

    def test_decode_filters_confidence_and_suppresses_overlap(self):
        # Exported YOLO output is [batch, channels, anchors], with many more
        # anchors than channels. Six anchors keep that real orientation here.
        output = np.zeros((1, 5, 6), dtype=np.float32)
        output[0, :, 0] = [100.0, 180.0, 80.0, 160.0, 0.90]
        output[0, :, 1] = [102.0, 182.0, 80.0, 160.0, 0.80]
        output[0, :, 2] = [400.0, 300.0, 50.0, 100.0, 0.10]
        transform = {
            "scale": 1.0,
            "pad_x": 0,
            "pad_y": 80,
            "original_width": 640,
            "original_height": 480,
        }

        detections = decode_yolo_output(output, transform, 0.35, 0.45)

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0]["confidence"], 0.90, places=5)
        self.assertEqual(detections[0]["track_id"], None)
        np.testing.assert_allclose(
            detections[0]["box"],
            [60.0, 20.0, 140.0, 180.0],
        )


if __name__ == "__main__":
    unittest.main()
