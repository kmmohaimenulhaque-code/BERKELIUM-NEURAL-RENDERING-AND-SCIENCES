"""Endpoint failures must become GatewayError / clean aborts, never tracebacks or fake 0 % baselines."""
import json
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from berkelium.ai.evaluate import EndpointLost, evaluate
from berkelium.ai.gateway import GatewayError, OpenAICompatibleProvider
from berkelium.ai.orchestrator import Orchestrator
from berkelium.cli import main


def _resetting_server():
    """Accepts, then closes with SO_LINGER=0 -> client sees ConnectionResetError (what the MI300X log showed)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            c.recv(65536)
            c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            c.close()
    threading.Thread(target=loop, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.getsockname()[1]}/v1"


def _models_server(ids):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": i} for i in ids]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def test_connection_reset_is_gateway_error():
    srv, url = _resetting_server()
    try:
        with pytest.raises(GatewayError, match="(?i)reset|RemoteDisconnected|BadStatusLine"):
            OpenAICompatibleProvider(url, "m").generate([{"role": "user", "content": "x"}])
    finally:
        srv.close()


def test_loopback_bypasses_proxy(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")   # dead proxy: would fail if used
    monkeypatch.setenv("no_proxy", "")
    srv, url = _models_server(["Qwen/Qwen3-32B"])
    try:
        assert OpenAICompatibleProvider(url, "Qwen/Qwen3-32B").check() == ["Qwen/Qwen3-32B"]
    finally:
        srv.shutdown()


def test_check_rejects_wrong_model_and_down_endpoint():
    srv, url = _models_server(["Qwen/Qwen3-32B"])
    try:
        with pytest.raises(GatewayError, match="does not serve 'berkelium'"):
            OpenAICompatibleProvider(url, "berkelium").check()
    finally:
        srv.shutdown()
    with pytest.raises(GatewayError):
        OpenAICompatibleProvider("http://127.0.0.1:1/v1", "m").check()


def test_eval_aborts_instead_of_scoring_zero():
    srv, url = _resetting_server()
    ex = [{"id": f"e{i}", "task": "plan", "split": "test", "family_id": "f",
           "messages": [{"role": "user", "content": "gear pair"}],
           "target": {"structure": {"components": []}}} for i in range(5)]
    try:
        with pytest.raises(EndpointLost):
            evaluate(ex, Orchestrator(OpenAICompatibleProvider(url, "m")), out=None)
    finally:
        srv.close()


def test_cli_eval_preflight_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("BERKELIUM_MODEL_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("BERKELIUM_MODEL", "Qwen/Qwen3-32B")
    out = tmp_path / "base.json"
    assert main(["eval", "--dataset", str(tmp_path), "--out", str(out)]) == 3
    assert not out.exists()
    assert main(["model-check"]) == 3
