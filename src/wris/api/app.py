from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from wris.api.schemas import PredictRequest
from wris.config.settings import load_settings
from wris.services.inference import RiskService

settings = load_settings()
service: RiskService | None = None
STATIC_DIR = Path(__file__).resolve().parent / "static"


def get_service() -> RiskService:
    global service
    if service is None:
        if not settings.artifacts.model_path.exists():
            raise RuntimeError("Model artifact not found. Run: python scripts/train_model.py")
        service = RiskService(settings.artifacts.model_path)
    return service


class WRISRequestHandler(BaseHTTPRequestHandler):
    def _write_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json({"status": "ok"})
            return

        if self.path in {"/", "/gui"}:
            index_file = STATIC_DIR / "index.html"
            if index_file.exists():
                self._write_html(index_file.read_text(encoding="utf-8"))
            else:
                self._write_json({"error": "GUI file not found"}, status=HTTPStatus.NOT_FOUND)
            return

        self._write_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._write_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8"))
            req = PredictRequest.from_dict(payload)
            out = get_service().predict(req.to_feature_dict())
            self._write_json(
                {
                    "burned_area_estimate": out.burned_area_estimate,
                    "risk_score": out.risk_score,
                    "risk_level": out.risk_level,
                }
            )
        except ValueError as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError:
            self._write_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = HTTPServer((host, port), WRISRequestHandler)
    print(f"WRIS API + GUI serving on http://{host}:{port}")
    server.serve_forever()
