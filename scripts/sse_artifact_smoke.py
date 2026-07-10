#!/usr/bin/env python3
"""Ephemeral isolated smoke: fake upstream + llama_proxy on 19052. Not a pytest test."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv/bin/python"


def make_request_body(artifact_id: str, parent_run: str, tmp_suffix: str) -> dict:
    return {
        "model": "test",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Artifact: {artifact_id}\n"
                    f"parent_run_id: {parent_run}\n"
                    f"Current working directory: /var/tmp/test-subagent-retrieve-{tmp_suffix}"
                ),
            }
        ],
    }


def run_smoke() -> int:
    aid1 = "agent_0123456789abcdef"
    aid2 = "agent_fedcba9876543210"
    cache_dir = tempfile.mkdtemp(prefix="llama-proxy-smoke-")
    fake_port = _free_port()
    proxy_port = _free_port()

    fake_code = (
        """
import json, time
from http.server import BaseHTTPRequestHandler, HTTPServer
AID = None
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        global AID
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        text = body["messages"][0]["content"]
        for token in text.replace("\\n", " ").split():
            if token.startswith("agent_"):
                AID = token
                break
        suffix = AID[6:] if AID else ""
        c1 = {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"agent_retrieve","arguments":'{"artifact_id":"agen'}}]}}]}
        c2 = {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"t_" + suffix + '"}'}}]}}]}
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for chunk in (c1, c2):
            self.wfile.write(("data: " + json.dumps(chunk) + "\\n\\n").encode())
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\\n\\n")
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", %d), H).serve_forever()
"""
        % fake_port
    )

    env = os.environ.copy()
    env.update(
        {
            "LLAMA_PROXY_UPSTREAM": f"http://127.0.0.1:{fake_port}",
            "LLAMA_PROXY_PORT": str(proxy_port),
            "LLAMA_PROXY_HOST": "127.0.0.1",
            "LLAMA_PROXY_CACHE_DIR": cache_dir,
            "LLAMA_PROXY_REPLAY_TPS": "0",
            "LLAMA_PROXY_REPLAY_CHUNK_DELAY_MS": "0",
        }
    )

    fake = subprocess.Popen(
        [str(PYTHON), "-c", fake_code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    proxy = subprocess.Popen(
        [str(PYTHON), "-m", "llama_proxy"],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(1.2)
        url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
        b1 = make_request_body(aid1, "run-aaa", "xyz")
        b2 = make_request_body(aid2, "run-bbb", "abc")

        with httpx.Client(timeout=30.0) as client:
            with client.stream("POST", url, json=b1) as r1:
                r1.raise_for_status()
                h1 = r1.headers.get("x-llama-proxy-cache", "miss")
                body1 = "".join(r1.iter_text())

            with client.stream("POST", url, json=b2) as r2:
                r2.raise_for_status()
                h2 = r2.headers.get("x-llama-proxy-cache", "miss")
                body2 = "".join(r2.iter_text())

        ok_miss1 = h1 != "hit"
        ok_hit = h2 == "hit"
        ok_current = aid2 in body2
        ok_no_stale = aid1 not in body2
        print(f"fake_port={fake_port} proxy_port={proxy_port}")
        print(f"req1_cache={h1}")
        print(f"req2_cache={h2}")
        print(f"miss1_ok={ok_miss1}")
        print(f"hit_ok={ok_hit}")
        print(f"current_id_in_response={ok_current}")
        print(f"no_stale_id={ok_no_stale}")
        if not (ok_miss1 and ok_hit and ok_current and ok_no_stale):
            print("body2_snippet:", body2[:800])
            return 1
        print("isolated_smoke_ok")
        return 0
    finally:
        proxy.terminate()
        fake.terminate()
        try:
            proxy.wait(timeout=3)
            fake.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proxy.kill()
            fake.kill()


if __name__ == "__main__":
    sys.exit(run_smoke())