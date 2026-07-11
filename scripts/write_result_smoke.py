#!/usr/bin/env python3
"""Contract smoke for Hatfield write-tool success path templating (not pytest)."""
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
    path1 = "/var/tmp/hatfield-write-aaa/.hatfield/tmp/out/file1.txt"
    path2 = "/var/tmp/hatfield-write-bbb/.hatfield/tmp/out/file2.txt"
    line1 = f"Successfully wrote 42 bytes to {path1}"
    line2 = f"Successfully wrote 42 bytes to {path2}"
    text = f"tool ok\n{line1}\nmore\n{line2}"

    paths = app._extract_write_result_paths_from_text(text)
    assert paths == [path1, path2], paths

    templated = app._apply_write_result_path_placeholders_to_string(text, paths)
    assert "{{write_result_path_0}}" in templated and "{{write_result_path_1}}" in templated
    assert path1 not in templated and path2 not in templated

    assert app._extract_write_result_paths_from_text("no write marker") == []
    assert app._extract_write_result_paths_from_text("Wrote 1 bytes to /tmp/x") == []

    body1 = {
        "model": "m",
        "messages": [{"role": "tool", "content": line1}],
    }
    body2 = {
        "model": "m",
        "messages": [{"role": "tool", "content": line2}],
    }
    k1, _ = app._cache_key("POST", "/v1/chat/completions", "", json.dumps(body1).encode())
    k2, _ = app._cache_key("POST", "/v1/chat/completions", "", json.dumps(body2).encode())
    assert k1 == k2, (k1, k2)

    other = {
        "model": "m",
        "messages": [{"role": "user", "content": "read /etc/hosts please"}],
    }
    ko1, _ = app._cache_key("POST", "/v1/chat/completions", "", json.dumps(other).encode())
    ko2, _ = app._cache_key(
        "POST",
        "/v1/chat/completions",
        "",
        json.dumps(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "read /etc/passwd please"}],
            }
        ).encode(),
    )
    assert ko1 != ko2

    stored = json.dumps({"content": line1}).encode()
    templ = app._template_response_bytes_for_cache(
        stored, output_cap_paths=None, write_result_paths=[path1]
    )
    assert b"{{write_result_path_0}}" in templ
    replay = app._substitute_cached_response_bytes(
        templ,
        artifact_id=None,
        output_cap_paths=[],
        write_result_paths=[path2],
    )
    assert replay is not None and path2.encode() in replay and path1.encode() not in replay

    missing = app._substitute_cached_response_bytes(
        b"{{write_result_path_0}}",
        artifact_id=None,
        output_cap_paths=[],
        write_result_paths=[],
    )
    assert missing is None

    sse_path = path1
    msg = line1
    half = len(msg) // 2
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
                                    "name": "write",
                                    "arguments": '{"note":"' + msg[:half],
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
                                    "arguments": msg[half:] + '"}',
                                },
                            }
                        ]
                    },
                }
            ]
        }
    )
    sse = f"data: {part1}\n\ndata: {part2}\n\ndata: [DONE]\n\n".encode()
    templ_sse = app._template_response_bytes_for_cache(
        sse, output_cap_paths=None, write_result_paths=[sse_path]
    )
    assert b"{{write_result_path_0}}" in templ_sse
    replay_sse = app._substitute_cached_response_bytes(
        templ_sse,
        artifact_id=None,
        output_cap_paths=[],
        write_result_paths=[path2],
    )
    assert replay_sse is not None and path2.encode() in replay_sse

    os.environ["LLAMA_PROXY_CACHE_TEMPLATE_WRITE_RESULT_PATHS"] = "false"
    app_off = importlib.reload(_load_app_module())
    k_off1, _ = app_off._cache_key(
        "POST", "/v1/chat/completions", "", json.dumps(body1).encode()
    )
    k_off2, _ = app_off._cache_key(
        "POST", "/v1/chat/completions", "", json.dumps(body2).encode()
    )
    assert k_off1 != k_off2
    os.environ.pop("LLAMA_PROXY_CACHE_TEMPLATE_WRITE_RESULT_PATHS", None)
    importlib.reload(_load_app_module())


def test_proxy_integration() -> None:
    path1 = "/var/tmp/hatfield-int-a/.hatfield/tmp/out/w1.txt"
    path2 = "/var/tmp/hatfield-int-b/.hatfield/tmp/out/w2.txt"
    cache_dir = tempfile.mkdtemp(prefix="llama-proxy-write-")
    fake_port = _free_port()
    proxy_port = _free_port()
    upstream_seen: list[str] = []

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
            if " bytes to " in line and line.startswith("Successfully wrote "):
                path = line.split(" bytes to ", 1)[1].strip()
                break
        print("UPSTREAM_PATH", path, flush=True)
        content = "Successfully wrote 1 bytes to " + path
        resp = {{"choices":[{{"message":{{"content": content}}}}]}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", {fake_port}), H).serve_forever()
"""
    fake_proc = subprocess.Popen(
        [str(PYTHON), "-c", fake_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proxy_env = {
        **os.environ,
        "LLAMA_PROXY_UPSTREAM": f"http://127.0.0.1:{fake_port}",
        "LLAMA_PROXY_PORT": str(proxy_port),
        "LLAMA_PROXY_HOST": "127.0.0.1",
        "LLAMA_PROXY_CACHE_DIR": cache_dir,
        "LLAMA_PROXY_CACHE_TEMPLATE_WRITE_RESULT_PATHS": "true",
        "LLAMA_PROXY_CACHE_TEMPLATE_ARTIFACT_IDS": "false",
        "LLAMA_PROXY_CACHE_TEMPLATE_OUTPUT_CAP_PATHS": "false",
    }
    proxy_proc = subprocess.Popen(
        [str(PYTHON), "-m", "llama_proxy"],
        cwd=str(REPO),
        env=proxy_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            try:
                httpx.get(f"http://127.0.0.1:{proxy_port}/__llama_proxy/health", timeout=0.5)
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("proxy did not start")

        def post(path: str) -> tuple[str | None, str]:
            body = {
                "model": "m",
                "messages": [
                    {
                        "role": "tool",
                        "content": f"Successfully wrote 12 bytes to {path}",
                    }
                ],
            }
            with httpx.Client(timeout=30.0) as client:
                r = client.post(
                    f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                    json=body,
                )
            return r.headers.get("x-llama-proxy-cache"), r.text

        h1, t1 = post(path1)
        assert h1 in (None, "miss"), h1
        if fake_proc.stdout:
            line = fake_proc.stdout.readline()
            assert path1 in line, line
            upstream_seen.append(path1)

        h2, t2 = post(path2)
        assert h2 == "hit", h2
        assert path2 in t2 and path1 not in t2, (t1, t2)
    finally:
        proxy_proc.terminate()
        fake_proc.terminate()
        proxy_proc.wait(timeout=5)
        fake_proc.wait(timeout=5)


def main() -> None:
    test_helpers()
    test_proxy_integration()
    print("write_result_smoke_ok")


if __name__ == "__main__":
    main()