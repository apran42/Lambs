"""HTTP entry point for the persistent Python 3.6 TensorRT worker."""

import argparse
import json
import struct
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, unquote, urlparse

from jetson_worker.service import TensorRTWorkerService, load_camera_definitions
from jetson_worker.tensorrt_engine import TensorRTEngine


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class WorkerHandler(BaseHTTPRequestHandler):
    def _write(self, status, payload, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _json(self, status, value):
        self._write(
            status,
            json.dumps(value, separators=(",", ":")).encode("utf-8"),
        )

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health" or parsed.path == "/api/cameras":
            self._json(200, self.server.worker.health())
            return
        prefix = "/api/packet/"
        if not parsed.path.startswith(prefix):
            self._json(404, {"detail": "Not found"})
            return
        camera_id = unquote(parsed.path[len(prefix):])
        query = parse_qs(parsed.query)
        try:
            after_frame_id = int(query.get("after_frame_id", [0])[0])
            timeout_ms = min(10000, max(0, int(query.get("timeout_ms", [2000])[0])))
            packet = self.server.worker.packet(
                camera_id, after_frame_id, timeout_ms / 1000.0
            )
        except KeyError:
            self._json(404, {"detail": "Unknown camera id"})
            return
        except (TypeError, ValueError) as exc:
            self._json(400, {"detail": str(exc)})
            return
        if packet is None:
            self._write(204, b"")
            return
        metadata, image_bytes = packet
        metadata_bytes = json.dumps(
            metadata, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        payload = struct.pack("!I", len(metadata_bytes)) + metadata_bytes + image_bytes
        self._write(200, payload, "application/octet-stream")


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    return parser.parse_args()


def main():
    args = _arguments()
    engine = TensorRTEngine(args.engine)
    worker = TensorRTWorkerService(
        load_camera_definitions(args.cameras),
        engine,
        confidence=args.confidence,
        iou=args.iou,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
    )
    worker.start(args.warmup)
    server = ThreadingHTTPServer((args.host, args.port), WorkerHandler)
    server.worker = worker
    print("TensorRT worker listening on http://{}:{}".format(args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.stop()
        engine.close()


if __name__ == "__main__":
    main()
