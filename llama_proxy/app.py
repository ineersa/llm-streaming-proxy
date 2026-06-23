from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

RESPONSE_STRIP_HEADERS = HOP_BY_HOP_HEADERS | {
    # httpx gives us decoded bytes when using aiter_bytes()/content.
    "content-encoding",
}

DEFAULT_CACHE_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/completion",
}

USER_CONTEXT_PREFIX = "[user-context] "


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    upstream: str
    cache_dir: Path
    cache_paths: set[str]
    host: str
    port: int
    request_timeout_s: float | None
    replay_tokens_per_second: float
    replay_chunk_delay_ms: float
    admin_token: str | None
    cache_normalize_messages: bool

    @classmethod
    def from_env(cls) -> "Settings":
        cache_paths = {
            p.strip()
            for p in os.getenv(
                "LLAMA_PROXY_CACHE_PATHS",
                ",".join(sorted(DEFAULT_CACHE_PATHS)),
            ).split(",")
            if p.strip()
        }
        timeout = os.getenv("LLAMA_PROXY_REQUEST_TIMEOUT_S", "600")
        return cls(
            upstream=os.getenv("LLAMA_PROXY_UPSTREAM", "http://127.0.0.1:8052").rstrip("/"),
            cache_dir=Path(os.getenv("LLAMA_PROXY_CACHE_DIR", "./llama_proxy_cache")).expanduser(),
            cache_paths=cache_paths,
            host=os.getenv("LLAMA_PROXY_HOST", "0.0.0.0"),
            port=int(os.getenv("LLAMA_PROXY_PORT", "9052")),
            request_timeout_s=None if timeout.lower() in {"", "none", "null", "0"} else float(timeout),
            replay_tokens_per_second=float(os.getenv("LLAMA_PROXY_REPLAY_TPS", "1000")),
            replay_chunk_delay_ms=float(os.getenv("LLAMA_PROXY_REPLAY_CHUNK_DELAY_MS", "0")),
            admin_token=os.getenv("LLAMA_PROXY_ADMIN_TOKEN") or None,
            cache_normalize_messages=_env_bool("LLAMA_PROXY_CACHE_NORMALIZE_MESSAGES", True),
        )


settings = Settings.from_env()
app = FastAPI(title="llama-proxy", version="0.1.0")
_client: httpx.AsyncClient | None = None
_locks: dict[str, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_request_json(body: bytes) -> Any | None:
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            return content["text"]
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") == "text" and isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _is_user_context_user_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    return _extract_text(message.get("content")).startswith(USER_CONTEXT_PREFIX)


def _strip_prologue_messages(messages: list[Any]) -> list[Any]:
    if not messages:
        return messages
    start = 0
    while start < len(messages):
        msg = messages[start]
        if not isinstance(msg, dict):
            break
        role = msg.get("role")
        if role in ("system", "developer"):
            start += 1
            continue
        if role == "user" and _is_user_context_user_message(msg):
            start += 1
            continue
        break
    return messages[start:]


def _normalize_body_for_cache_key(body: Any) -> Any:
    if not isinstance(body, dict):
        return body
    if not settings.cache_normalize_messages:
        return body
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    normalized = copy.deepcopy(body)
    normalized["messages"] = _strip_prologue_messages(messages)
    return normalized


def _cache_key(method: str, path: str, query: str, body: bytes) -> tuple[str, dict[str, Any]]:
    parsed_json = _safe_request_json(body)
    if parsed_json is None:
        body_for_key: Any = {"raw_body_b64": base64.b64encode(body).decode("ascii")}
        request_material: dict[str, Any] = {
            "method": method.upper(),
            "path": path,
            "query": query,
            "body": body_for_key,
        }
    else:
        body_for_key = _normalize_body_for_cache_key(parsed_json)
        request_material = {
            "method": method.upper(),
            "path": path,
            "query": query,
            "body": parsed_json,
            "key_body": body_for_key,
        }

    key_material = {
        "method": method.upper(),
        "path": path,
        "query": query,
        "body": body_for_key,
    }
    canonical = _json_dumps(key_material).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), request_material


def _cache_file(key: str) -> Path:
    return settings.cache_dir / key[:2] / f"{key}.json"


def _read_record(key: str) -> dict[str, Any] | None:
    path = _cache_file(key)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_record(key: str, record: dict[str, Any]) -> None:
    path = _cache_file(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_dumps(record)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _request_headers(request: Request) -> dict[str, str]:
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP_HEADERS
    }
    # Keep recorded bytes simple and avoid upstream compression surprises.
    headers["accept-encoding"] = "identity"
    return headers


def _response_headers(headers: httpx.Headers | dict[str, str]) -> dict[str, str]:
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in RESPONSE_STRIP_HEADERS
    }


def _upstream_url(path: str, query: str) -> str:
    url = f"{settings.upstream}/{path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"
    return url


def _is_cacheable(method: str, path: str) -> bool:
    return method.upper() == "POST" and path in settings.cache_paths


def _is_stream_request(body: bytes) -> bool:
    parsed = _safe_request_json(body)
    return isinstance(parsed, dict) and parsed.get("stream") is True


def _body_b64(body: bytes) -> str:
    return base64.b64encode(body).decode("ascii")


def _from_b64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def _replay_delay_s(chunk: bytes) -> float:
    delay = max(0.0, settings.replay_chunk_delay_ms / 1000.0)
    if settings.replay_tokens_per_second > 0:
        # Approximate: 1 token ~= 4 bytes. This keeps replay deterministic and fast;
        # it is intentionally not trying to reproduce upstream latency.
        delay = max(delay, (len(chunk) / 4.0) / settings.replay_tokens_per_second)
    return delay


async def _lock_for(key: str) -> asyncio.Lock:
    async with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _locks[key] = lock
        return lock


def _check_admin(request: Request) -> None:
    if settings.admin_token and request.headers.get("x-llama-proxy-token") != settings.admin_token:
        raise HTTPException(status_code=403, detail="invalid admin token")


@app.on_event("startup")
async def startup() -> None:
    global _client
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(settings.request_timeout_s) if settings.request_timeout_s else None
    _client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)


@app.on_event("shutdown")
async def shutdown() -> None:
    if _client is not None:
        await _client.aclose()


@app.get("/__llama_proxy/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "upstream": settings.upstream,
        "cache_dir": str(settings.cache_dir),
        "cache_paths": sorted(settings.cache_paths),
        "cache_normalize_messages": settings.cache_normalize_messages,
    }


@app.get("/__llama_proxy/cache/stats")
async def cache_stats(request: Request) -> dict[str, Any]:
    _check_admin(request)
    files = list(settings.cache_dir.glob("*/*.json")) if settings.cache_dir.exists() else []
    return {
        "entries": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "cache_dir": str(settings.cache_dir),
    }


@app.post("/__llama_proxy/cache/clear")
@app.delete("/__llama_proxy/cache")
async def clear_cache(request: Request) -> dict[str, Any]:
    _check_admin(request)
    if settings.cache_dir.exists():
        shutil.rmtree(settings.cache_dir)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "cache_dir": str(settings.cache_dir)}


def _outbound_headers(record: dict[str, Any], cache_marker: str | None = None) -> dict[str, str]:
    headers = dict(record.get("upstream", {}).get("headers", {}))
    if cache_marker:
        headers["x-llama-proxy-cache"] = cache_marker
    return headers


async def _replay(record: dict[str, Any], *, cache_marker: str | None = None) -> Response:
    status_code = int(record.get("upstream", {}).get("status_code", 200))
    headers = _outbound_headers(record, cache_marker)

    if record.get("stream"):
        chunks = [_from_b64(item) for item in record.get("chunks_b64", [])]

        async def generate():
            for chunk in chunks:
                yield chunk
                delay = _replay_delay_s(chunk)
                if delay:
                    await asyncio.sleep(delay)

        return StreamingResponse(generate(), status_code=status_code, headers=headers)

    return Response(
        content=_from_b64(record.get("body_b64", "")),
        status_code=status_code,
        headers=headers,
    )


async def _proxy_uncached(request: Request, path: str, body: bytes) -> Response:
    assert _client is not None
    query = request.url.query
    upstream = await _client.request(
        request.method,
        _upstream_url(path, query),
        content=body,
        headers=_request_headers(request),
    )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=_response_headers(upstream.headers),
    )


async def _proxy_and_record_nonstream(
    request: Request,
    path: str,
    body: bytes,
    key: str,
    request_material: dict[str, Any],
) -> Response:
    assert _client is not None
    started = time.monotonic()
    upstream = await _client.request(
        request.method,
        _upstream_url(path, request.url.query),
        content=body,
        headers=_request_headers(request),
    )
    headers = _response_headers(upstream.headers)

    if 200 <= upstream.status_code < 300:
        _write_record(
            key,
            {
                "key": key,
                "created_at": time.time(),
                "request": request_material,
                "stream": False,
                "upstream": {
                    "status_code": upstream.status_code,
                    "headers": headers,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                },
                "body_b64": _body_b64(upstream.content),
            },
        )

    return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)


async def _proxy_and_record_stream(
    request: Request,
    path: str,
    body: bytes,
    key: str,
    request_material: dict[str, Any],
    lock: asyncio.Lock,
) -> StreamingResponse:
    assert _client is not None
    started = time.monotonic()
    cm = _client.stream(
        request.method,
        _upstream_url(path, request.url.query),
        content=body,
        headers=_request_headers(request),
    )
    upstream = await cm.__aenter__()
    headers = _response_headers(upstream.headers)

    async def generate():
        chunks: list[bytes] = []
        completed = False
        try:
            async for chunk in upstream.aiter_bytes():
                chunks.append(chunk)
                yield chunk
            completed = True
        finally:
            await cm.__aexit__(None, None, None)
            if completed and 200 <= upstream.status_code < 300:
                _write_record(
                    key,
                    {
                        "key": key,
                        "created_at": time.time(),
                        "request": request_material,
                        "stream": True,
                        "upstream": {
                            "status_code": upstream.status_code,
                            "headers": headers,
                            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                        },
                        "chunks_b64": [_body_b64(chunk) for chunk in chunks],
                    },
                )
            lock.release()

    return StreamingResponse(generate(), status_code=upstream.status_code, headers=headers)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str) -> Response:
    path = f"/{path}"
    body = await request.body()

    if not _is_cacheable(request.method, path):
        return await _proxy_uncached(request, path, body)

    key, material = _cache_key(request.method, path, request.url.query, body)
    cached = _read_record(key)
    if cached is not None:
        return await _replay(cached, cache_marker="hit")

    lock = await _lock_for(key)
    await lock.acquire()
    stream_request = _is_stream_request(body)
    try:
        cached = _read_record(key)
        if cached is not None:
            lock.release()
            return await _replay(cached, cache_marker="hit")

        if stream_request:
            try:
                # The StreamingResponse generator owns lock release after it has saved the cassette.
                return await _proxy_and_record_stream(request, path, body, key, material, lock)
            except Exception:
                if lock.locked():
                    lock.release()
                raise

        return await _proxy_and_record_nonstream(request, path, body, key, material)
    finally:
        if lock.locked() and not stream_request:
            lock.release()


def main() -> None:
    uvicorn.run("llama_proxy.app:app", host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
