from __future__ import annotations

import json
import socket
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from wris.api.schemas import PredictRequest
from wris.config.settings import load_settings
from wris.data.ingestion import download_csv, load_rows
from wris.models.training import save_artifacts, train_and_evaluate
from wris.services.inference import RiskService

settings = load_settings()
service: RiskService | None = None
STATIC_DIR = Path(__file__).resolve().parent / "static"


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) != 0


def _find_open_port(host: str, preferred: int, max_tries: int = 20) -> int:
    for candidate in range(preferred, preferred + max_tries):
        if _is_port_available(host, candidate):
            return candidate
    raise RuntimeError(
        f"No available port found in range {preferred}-{preferred + max_tries - 1}."
    )


def _bootstrap_model_if_missing() -> None:
    if settings.artifacts.model_path.exists():
        return

    download_csv(settings.data.url, settings.data.raw_path)
    rows = load_rows(settings.data.raw_path)
    model, metrics = train_and_evaluate(
        rows=rows,
        target=settings.model.target,
        test_size=settings.model.test_size,
        random_state=settings.model.random_state,
        k_neighbors=settings.model.k_neighbors,
    )
    save_artifacts(
        model=model,
        metrics=metrics,
        model_path=settings.artifacts.model_path,
        metrics_path=settings.artifacts.metrics_path,
    )


def get_service() -> RiskService:
    global service
    if service is None:
        _bootstrap_model_if_missing()
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
    chosen_port = _find_open_port(host, port)
    if chosen_port != port:
        print(f"Port {port} is busy. Using available port {chosen_port} instead.")

    server = ReusableHTTPServer((host, chosen_port), WRISRequestHandler)
    print(f"WRIS API + GUI serving on http://{host}:{chosen_port}")
    server.serve_forever()
