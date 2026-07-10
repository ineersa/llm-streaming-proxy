#!/usr/bin/env python3
"""Contract smoke for output-cap path templating (not pytest)."""
from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv/bin/python"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _load_app_module():
    sys.path.insert(0, str(REPO))
    import llama_proxy.app as app_module

    return importlib.reload(app_module)


def test_helpers() -> None:
    app = _load_app_module()
    path1 = "/tmp/a/.hatfield/tmp/output-cap/20260101-abc123.txt"
    path2 = "/tmp/b/.hatfield/tmp/output-cap/20260102-def456.txt"
    text = (
        f"[Output capped: 1 chars]\nSaved full output: {path1}\n"
        f"read({path1})\nSaved full output: {path2}"
    )
    paths = app._extract_output_cap_paths_from_text(text)
    assert paths == [path1, path2], paths
    templated = app._apply_output_cap_path_placeholders_to_string(text, paths)
    assert "{{output_cap_path_0}}" in templated and "{{output_cap_path_1}}" in templated
    assert path1 not in templated and path2 not in templated
    unrelated = "Saved full output: /etc/passwd is not this"
    assert app._extract_output_cap_paths_from_text("no marker here") == []
    assert app._extract_output_cap_paths_from_text(unrelated) == ["/etc/passwd is not this"]

    body1 = {
        "model": "m",
        "messages": [{"role": "user", "content": f"Saved full output: {path1}"}],
    }
    body2 = {
        "model": "m",
        "messages": [{"role": "user", "content": f"Saved full output: {path2}"}],
    }
    k1, _ = app._cache_key("POST", "/v1/chat/completions", "", json.dumps(body1).encode())
    k2, _ = app._cache_key("POST", "/v1/chat/completions", "", json.dumps(body2).encode())
    assert k1 == k2, (k1, k2)

    no_marker = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    kn, mat = app._cache_key("POST", "/v1/chat/completions", "", json.dumps(no_marker).encode())
    assert "hello" in json.dumps(mat["key_body"])

    other_path_body = {
        "model": "m",
        "messages": [{"role": "user", "content": "Use /var/lib/foo/bar.txt please"}],
    }
    ko1, _ = app._cache_key(
        "POST", "/v1/chat/completions", "", json.dumps(other_path_body).encode()
    )
    ko2, _ = app._cache_key(
        "POST",
        "/v1/chat/completions",
        "",
        json.dumps(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "Use /var/lib/foo/baz.txt please"}],
            }
        ).encode(),
    )
    assert ko1 != ko2

    stored = f'{{"path":"{path1}"}}'.encode()
    templ = app._template_response_bytes_for_cache(stored, output_cap_paths=[path1])
    assert b"{{output_cap_path_0}}" in templ
    replay = app._substitute_cached_response_bytes(
        templ, artifact_id=None, output_cap_paths=[path2]
    )
    assert replay is not None and path2.encode() in replay and path1.encode() not in replay

    missing = app._substitute_cached_response_bytes(
        b"{{output_cap_path_0}}", artifact_id=None, output_cap_paths=[]
    )
    assert missing is None

    sse_path = path1
    part1 = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "read",
                                    "arguments": '{"path":"' + sse_path[: len(sse_path) // 2],
                                },
                            }
                        ]
                    },
                }
            ]
        }
    )
    part2 = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": sse_path[len(sse_path) // 2 :] + '"}',
                                },
                            }
                        ]
                    },
                }
            ]
        }
    )
    sse = f"data: {part1}\n\ndata: {part2}\n\ndata: [DONE]\n\n".encode()
    templ_sse = app._template_response_bytes_for_cache(sse, output_cap_paths=[sse_path])
    assert b"{{output_cap_path_0}}" in templ_sse
    replay_sse = app._substitute_cached_response_bytes(
        templ_sse, artifact_id=None, output_cap_paths=[path2]
    )
    assert replay_sse is not None and path2.encode() in replay_sse

    os.environ["LLAMA_PROXY_CACHE_TEMPLATE_OUTPUT_CAP_PATHS"] = "false"
    app_off = importlib.reload(_load_app_module())
    k_off1, _ = app_off._cache_key(
        "POST", "/v1/chat/completions", "", json.dumps(body1).encode()
    )
    k_off2, _ = app_off._cache_key(
        "POST", "/v1/chat/completions", "", json.dumps(body2).encode()
    )
    assert k_off1 != k_off2
    os.environ.pop("LLAMA_PROXY_CACHE_TEMPLATE_OUTPUT_CAP_PATHS", None)
    importlib.reload(_load_app_module())


def test_proxy_integration() -> None:
    path1 = "/var/tmp/test-output-cap-aaa/.hatfield/tmp/output-cap/20260101-1111111111111111.txt"
    path2 = "/var/tmp/test-output-cap-bbb/.hatfield/tmp/output-cap/20260102-2222222222222222.txt"
    cache_dir = tempfile.mkdtemp(prefix="llama-proxy-outcap-")
    fake_port = _free_port()
    proxy_port = _free_port()

    fake_code = f"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        text = body["messages"][0]["content"]
        path = ""
        for line in text.splitlines():
            if line.startswith("Saved full output:"):
                path = line.split(":", 1)[1].strip()
                break
        payload = {{"choices":[{{"message":{{"content":"Use " + path}}}}]}}
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", {fake_port}), H).serve_forever()
"""
    fake = subprocess.Popen([str(PYTHON), "-c", fake_code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    env = os.environ.copy()
    env.update(
        {
            "LLAMA_PROXY_UPSTREAM": f"http://127.0.0.1:{fake_port}",
            "LLAMA_PROXY_HOST": "127.0.0.1",
            "LLAMA_PROXY_PORT": str(proxy_port),
            "LLAMA_PROXY_CACHE_DIR": cache_dir,
            "LLAMA_PROXY_CACHE_TEMPLATE_OUTPUT_CAP_PATHS": "true",
            "LLAMA_PROXY_CACHE_TEMPLATE_ARTIFACT_IDS": "false",
        }
    )
    proxy = subprocess.Popen(
        [str(PYTHON), "-m", "llama_proxy"],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            try:
                httpx.get(f"http://127.0.0.1:{proxy_port}/__llama_proxy/health", timeout=0.5)
                break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            raise RuntimeError("proxy did not start")

        def body_for(path: str) -> dict:
            return {
                "model": "test",
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": f"[Output capped: 1 chars]\nSaved full output: {path}",
                    }
                ],
            }

        with httpx.Client(timeout=30.0) as client:
            r1 = client.post(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                json=body_for(path1),
            )
            assert r1.headers.get("x-llama-proxy-cache") in (None, "miss"), r1.headers
            assert path1 in r1.text

            r2 = client.post(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                json=body_for(path2),
            )
            assert r2.headers.get("x-llama-proxy-cache") == "hit", r2.headers
            assert path2 in r2.text
            assert path1 not in r2.text
    finally:
        proxy.terminate()
        fake.terminate()
        proxy.wait(timeout=5)
        fake.wait(timeout=5)


def main() -> int:
    test_helpers()
    test_proxy_integration()
    print("output_cap_smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())