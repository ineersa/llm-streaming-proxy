from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
import re
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
AGENT_ARTIFACT_ID_RE = re.compile(r"\bagent_[0-9a-f]{16}\b")
AGENT_ARTIFACT_PLACEHOLDER = "{{agent_artifact_id}}"
ARTIFACT_LABEL = "Artifact:"
PARENT_RUN_ID_LABEL_RE = re.compile(r"parent_run_id:\s*[A-Za-z0-9_-]+")
PARENT_RUN_ID_JSON_RE = re.compile(r'"parent_run_id"\s*:\s*"[^"]+"')
AGENT_RUN_ID_LABEL_RE = re.compile(r"agent_run_id:\s*[0-9a-fA-F-]{36}")
AGENT_RUN_ID_JSON_RE = re.compile(r'"agent_run_id"\s*:\s*"[0-9a-fA-F-]{36}"')
TEST_SUBAGENT_TMP_PATH_RE = re.compile(r"/var/tmp/test-subagent-retrieve-[A-Za-z0-9_-]+")
TEST_SUBAGENT_CWD_RE = re.compile(
    r"Current working directory:\s*/var/tmp/test-subagent-retrieve-[^\s]+"
)
ARTIFACT_ID_LABEL_RE = re.compile(r"Artifact ID:\s*agent_[0-9a-f]{16}")
OUTPUT_CAP_SAVED_LINE_PREFIX = "Saved full output:"
OUTPUT_CAP_PATH_PLACEHOLDER_RE = re.compile(r"\{\{output_cap_path_(\d+)\}\}")
WRITE_RESULT_SUCCESS_RE = re.compile(r"Successfully wrote \d+ bytes to ([^\n]+)")
WRITE_RESULT_PATH_PLACEHOLDER_RE = re.compile(r"\{\{write_result_path_(\d+)\}\}")
TOOL_RESULT_IMAGE_PATH_RE = re.compile(r"\[Tool result image:\s*([^\s(]+)")
VIEW_IMAGE_PATH_PLACEHOLDER_RE = re.compile(r"\{\{view_image_path_(\d+)\}\}")
PARENT_RUN_ID_PLACEHOLDER = "{{parent_run_id}}"
AGENT_RUN_ID_PLACEHOLDER = "{{agent_run_id}}"
TEST_SUBAGENT_TMP_PLACEHOLDER = "{{test_subagent_tmp}}"


def _output_cap_path_placeholder(index: int) -> str:
    return f"{{{{output_cap_path_{index}}}}}"


def _write_result_path_placeholder(index: int) -> str:
    return f"{{{{write_result_path_{index}}}}}"


def _view_image_path_placeholder(index: int) -> str:
    return f"{{{{view_image_path_{index}}}}}"


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
    cache_template_artifact_ids: bool
    cache_template_output_cap_paths: bool
    cache_template_write_result_paths: bool
    cache_template_view_image_paths: bool

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
            cache_template_artifact_ids=_env_bool("LLAMA_PROXY_CACHE_TEMPLATE_ARTIFACT_IDS", True),
            cache_template_output_cap_paths=_env_bool(
                "LLAMA_PROXY_CACHE_TEMPLATE_OUTPUT_CAP_PATHS", True
            ),
            cache_template_write_result_paths=_env_bool(
                "LLAMA_PROXY_CACHE_TEMPLATE_WRITE_RESULT_PATHS", True
            ),
            cache_template_view_image_paths=_env_bool(
                "LLAMA_PROXY_CACHE_TEMPLATE_VIEW_IMAGE_PATHS", True
            ),
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


def _normalize_body_for_cache_key_messages(body: Any) -> Any:
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


def _collect_strings(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, out)


def _extract_agent_artifact_id_from_text(text: str) -> str | None:
    if not text:
        return None
    after_label = text.split(ARTIFACT_LABEL, 1)
    if len(after_label) == 2:
        match = AGENT_ARTIFACT_ID_RE.search(after_label[1])
        if match:
            return match.group(0)
    match = AGENT_ARTIFACT_ID_RE.search(text)
    return match.group(0) if match else None


def _extract_agent_artifact_id_from_json(value: Any) -> str | None:
    strings: list[str] = []
    _collect_strings(value, strings)
    for text in strings:
        if ARTIFACT_LABEL in text:
            artifact_id = _extract_agent_artifact_id_from_text(text)
            if artifact_id:
                return artifact_id
    for text in strings:
        artifact_id = _extract_agent_artifact_id_from_text(text)
        if artifact_id:
            return artifact_id
    return None


def _extract_agent_artifact_id_from_bytes(body: bytes) -> str | None:
    parsed = _safe_request_json(body)
    if parsed is not None:
        return _extract_agent_artifact_id_from_json(parsed)
    if not body:
        return None
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return _extract_agent_artifact_id_from_text(text)


def _extract_output_cap_paths_from_text(text: str) -> list[str]:
    if not text:
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if OUTPUT_CAP_SAVED_LINE_PREFIX not in line:
            continue
        idx = line.find(OUTPUT_CAP_SAVED_LINE_PREFIX)
        path = line[idx + len(OUTPUT_CAP_SAVED_LINE_PREFIX) :].strip()
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _extract_output_cap_paths_from_json(value: Any) -> list[str]:
    strings: list[str] = []
    _collect_strings(value, strings)
    ordered: list[str] = []
    seen: set[str] = set()
    for text in strings:
        for path in _extract_output_cap_paths_from_text(text):
            if path not in seen:
                seen.add(path)
                ordered.append(path)
    return ordered


def _extract_output_cap_paths_from_bytes(body: bytes) -> list[str]:
    parsed = _safe_request_json(body)
    if parsed is not None:
        return _extract_output_cap_paths_from_json(parsed)
    if not body:
        return []
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return _extract_output_cap_paths_from_text(text)


def _apply_output_cap_path_placeholders_to_string(text: str, paths: list[str]) -> str:
    return _apply_indexed_path_placeholders_to_string(text, paths, _output_cap_path_placeholder)


def _template_output_cap_paths_in_value(value: Any, paths: list[str]) -> Any:
    return _template_indexed_paths_in_value(value, paths, _output_cap_path_placeholder)


def _template_output_cap_paths_in_bytes(data: bytes, paths: list[str]) -> bytes:
    return _template_indexed_paths_in_bytes(data, paths, _output_cap_path_placeholder)


def _substitute_output_cap_placeholders_in_string(text: str, paths: list[str]) -> str | None:
    return _substitute_indexed_placeholders_in_string(
        text, paths, OUTPUT_CAP_PATH_PLACEHOLDER_RE, _output_cap_path_placeholder
    )


def _substitute_output_cap_placeholders_in_bytes(data: bytes, paths: list[str]) -> bytes | None:
    return _substitute_indexed_placeholders_in_bytes(
        data, paths, OUTPUT_CAP_PATH_PLACEHOLDER_RE, _output_cap_path_placeholder
    )


def _extract_write_result_paths_from_text(text: str) -> list[str]:
    if not text:
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for match in WRITE_RESULT_SUCCESS_RE.finditer(text):
        path = match.group(1).strip()
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _extract_write_result_paths_from_json(value: Any) -> list[str]:
    strings: list[str] = []
    _collect_strings(value, strings)
    ordered: list[str] = []
    seen: set[str] = set()
    for text in strings:
        for path in _extract_write_result_paths_from_text(text):
            if path not in seen:
                seen.add(path)
                ordered.append(path)
    return ordered


def _extract_write_result_paths_from_bytes(body: bytes) -> list[str]:
    parsed = _safe_request_json(body)
    if parsed is not None:
        return _extract_write_result_paths_from_json(parsed)
    if not body:
        return []
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return _extract_write_result_paths_from_text(text)


def _apply_write_result_path_placeholders_to_string(text: str, paths: list[str]) -> str:
    return _apply_indexed_path_placeholders_to_string(text, paths, _write_result_path_placeholder)


def _template_write_result_paths_in_value(value: Any, paths: list[str]) -> Any:
    return _template_indexed_paths_in_value(value, paths, _write_result_path_placeholder)


def _template_write_result_paths_in_bytes(data: bytes, paths: list[str]) -> bytes:
    return _template_indexed_paths_in_bytes(data, paths, _write_result_path_placeholder)


def _substitute_write_result_placeholders_in_string(text: str, paths: list[str]) -> str | None:
    return _substitute_indexed_placeholders_in_string(
        text, paths, WRITE_RESULT_PATH_PLACEHOLDER_RE, _write_result_path_placeholder
    )


def _substitute_write_result_placeholders_in_bytes(data: bytes, paths: list[str]) -> bytes | None:
    return _substitute_indexed_placeholders_in_bytes(
        data, paths, WRITE_RESULT_PATH_PLACEHOLDER_RE, _write_result_path_placeholder
    )


def _first_json_object_from_text(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _append_view_image_paths_from_obj(obj: dict[str, Any], ordered: list[str], seen: set[str]) -> None:
    if obj.get("type") != "view_image":
        return
    top = obj.get("path")
    if isinstance(top, str):
        p = top.strip()
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)
    refs = obj.get("attachment_refs")
    if isinstance(refs, list):
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            rp = ref.get("path")
            if isinstance(rp, str):
                p = rp.strip()
                if p and p not in seen:
                    seen.add(p)
                    ordered.append(p)


def _extract_view_image_paths_from_text(text: str) -> list[str]:
    if not text:
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = _first_json_object_from_text(line)
        if obj is not None:
            _append_view_image_paths_from_obj(obj, ordered, seen)
    for match in TOOL_RESULT_IMAGE_PATH_RE.finditer(text):
        p = match.group(1).strip()
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _extract_view_image_paths_from_json(value: Any) -> list[str]:
    strings: list[str] = []
    _collect_strings(value, strings)
    ordered: list[str] = []
    seen: set[str] = set()
    for text in strings:
        for path in _extract_view_image_paths_from_text(text):
            if path not in seen:
                seen.add(path)
                ordered.append(path)
    return ordered


def _extract_view_image_paths_from_bytes(body: bytes) -> list[str]:
    parsed = _safe_request_json(body)
    if parsed is not None:
        return _extract_view_image_paths_from_json(parsed)
    if not body:
        return []
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return _extract_view_image_paths_from_text(text)


def _apply_view_image_path_placeholders_to_string(text: str, paths: list[str]) -> str:
    return _apply_indexed_path_placeholders_to_string(text, paths, _view_image_path_placeholder)


def _template_view_image_paths_in_value(value: Any, paths: list[str]) -> Any:
    return _template_indexed_paths_in_value(value, paths, _view_image_path_placeholder)


def _template_view_image_paths_in_bytes(data: bytes, paths: list[str]) -> bytes:
    return _template_indexed_paths_in_bytes(data, paths, _view_image_path_placeholder)


def _substitute_view_image_placeholders_in_string(text: str, paths: list[str]) -> str | None:
    return _substitute_indexed_placeholders_in_string(
        text, paths, VIEW_IMAGE_PATH_PLACEHOLDER_RE, _view_image_path_placeholder
    )


def _substitute_view_image_placeholders_in_bytes(data: bytes, paths: list[str]) -> bytes | None:
    return _substitute_indexed_placeholders_in_bytes(
        data, paths, VIEW_IMAGE_PATH_PLACEHOLDER_RE, _view_image_path_placeholder
    )


def _apply_indexed_path_placeholders_to_string(
    text: str,
    paths: list[str],
    placeholder_fn: Any,
) -> str:
    if not text or not paths:
        return text
    for index, path in enumerate(paths):
        if path:
            text = text.replace(path, placeholder_fn(index))
    return text


def _template_indexed_paths_in_value(
    value: Any,
    paths: list[str],
    placeholder_fn: Any,
) -> Any:
    if not paths:
        return value
    if isinstance(value, str):
        return _apply_indexed_path_placeholders_to_string(value, paths, placeholder_fn)
    if isinstance(value, dict):
        return {
            k: _template_indexed_paths_in_value(v, paths, placeholder_fn) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_template_indexed_paths_in_value(item, paths, placeholder_fn) for item in value]
    return value


def _template_indexed_paths_in_bytes(
    data: bytes,
    paths: list[str],
    placeholder_fn: Any,
) -> bytes:
    if not data or not paths:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return _apply_indexed_path_placeholders_to_string(text, paths, placeholder_fn).encode("utf-8")


def _substitute_indexed_placeholders_in_string(
    text: str,
    paths: list[str],
    placeholder_re: re.Pattern[str],
    placeholder_fn: Any,
) -> str | None:
    if not text:
        return text
    if not paths:
        if placeholder_re.search(text):
            return None
        return text
    max_index = -1
    for match in placeholder_re.finditer(text):
        max_index = max(max_index, int(match.group(1)))
    if max_index >= len(paths):
        return None
    for index, path in enumerate(paths):
        text = text.replace(placeholder_fn(index), path)
    if placeholder_re.search(text):
        return None
    return text


def _substitute_indexed_placeholders_in_bytes(
    data: bytes,
    paths: list[str],
    placeholder_re: re.Pattern[str],
    placeholder_fn: Any,
) -> bytes | None:
    if not data:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    substituted = _substitute_indexed_placeholders_in_string(text, paths, placeholder_re, placeholder_fn)
    if substituted is None:
        return None
    return substituted.encode("utf-8")


def _template_agent_artifact_ids_in_value(value: Any) -> Any:
    if isinstance(value, str):
        return AGENT_ARTIFACT_ID_RE.sub(AGENT_ARTIFACT_PLACEHOLDER, value)
    if isinstance(value, dict):
        return {k: _template_agent_artifact_ids_in_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_template_agent_artifact_ids_in_value(item) for item in value]
    return value


def _template_agent_artifact_ids_in_bytes(data: bytes) -> bytes:
    if not data:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return AGENT_ARTIFACT_ID_RE.sub(AGENT_ARTIFACT_PLACEHOLDER, text).encode("utf-8")


def _normalize_dynamic_request_fields_in_string(text: str) -> str:
    if not text:
        return text
    text = PARENT_RUN_ID_LABEL_RE.sub(f"parent_run_id: {PARENT_RUN_ID_PLACEHOLDER}", text)
    text = PARENT_RUN_ID_JSON_RE.sub(f'"parent_run_id":"{PARENT_RUN_ID_PLACEHOLDER}"', text)
    text = AGENT_RUN_ID_LABEL_RE.sub(f"agent_run_id: {AGENT_RUN_ID_PLACEHOLDER}", text)
    text = AGENT_RUN_ID_JSON_RE.sub(f'"agent_run_id":"{AGENT_RUN_ID_PLACEHOLDER}"', text)
    text = TEST_SUBAGENT_CWD_RE.sub(
        f"Current working directory: {TEST_SUBAGENT_TMP_PLACEHOLDER}", text
    )
    text = TEST_SUBAGENT_TMP_PATH_RE.sub(TEST_SUBAGENT_TMP_PLACEHOLDER, text)
    text = ARTIFACT_ID_LABEL_RE.sub(f"Artifact ID: {AGENT_ARTIFACT_PLACEHOLDER}", text)
    return text


def _normalize_dynamic_request_fields_in_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_dynamic_request_fields_in_string(value)
    if isinstance(value, dict):
        return {k: _normalize_dynamic_request_fields_in_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_dynamic_request_fields_in_value(item) for item in value]
    return value


def _template_tool_call_arguments_string(
    arguments: str,
    output_cap_paths: list[str] | None = None,
    write_result_paths: list[str] | None = None,
    view_image_paths: list[str] | None = None,
) -> str:
    if not arguments:
        return arguments
    templated = AGENT_ARTIFACT_ID_RE.sub(AGENT_ARTIFACT_PLACEHOLDER, arguments)
    if output_cap_paths:
        templated = _apply_output_cap_path_placeholders_to_string(templated, output_cap_paths)
    if write_result_paths:
        templated = _apply_write_result_path_placeholders_to_string(templated, write_result_paths)
    if view_image_paths:
        templated = _apply_view_image_path_placeholders_to_string(templated, view_image_paths)
    return templated


def _template_tool_calls_in_json_value(
    value: Any,
    output_cap_paths: list[str] | None = None,
    write_result_paths: list[str] | None = None,
    view_image_paths: list[str] | None = None,
) -> Any:
    cap_paths = output_cap_paths or []
    write_paths = write_result_paths or []
    view_paths = view_image_paths or []
    if isinstance(value, dict):
        out = {
            k: _template_tool_calls_in_json_value(
                v, output_cap_paths, write_result_paths, view_image_paths
            )
            for k, v in value.items()
        }
        if "tool_calls" in out and isinstance(out["tool_calls"], list):
            for tc in out["tool_calls"]:
                if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                    fn = tc["function"]
                    if isinstance(fn.get("arguments"), str):
                        fn["arguments"] = _template_tool_call_arguments_string(
                            fn["arguments"], cap_paths, write_paths, view_paths
                        )
        if "function_call" in out and isinstance(out["function_call"], dict):
            fc = out["function_call"]
            if isinstance(fc.get("arguments"), str):
                fc["arguments"] = _template_tool_call_arguments_string(
                    fc["arguments"], cap_paths, write_paths, view_paths
                )
        if "delta" in out and isinstance(out["delta"], dict):
            delta = out["delta"]
            if isinstance(delta.get("tool_calls"), list):
                for tc in delta["tool_calls"]:
                    if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                        fn = tc["function"]
                        if isinstance(fn.get("arguments"), str):
                            fn["arguments"] = _template_tool_call_arguments_string(
                                fn["arguments"], cap_paths, write_paths, view_paths
                            )
            if isinstance(delta.get("function_call"), dict):
                fc = delta["function_call"]
                if isinstance(fc.get("arguments"), str):
                    fc["arguments"] = _template_tool_call_arguments_string(
                        fc["arguments"], cap_paths, write_paths, view_paths
                    )
        return out
    if isinstance(value, list):
        return [
            _template_tool_calls_in_json_value(
                item, output_cap_paths, write_result_paths, view_image_paths
            )
            for item in value
        ]
    return value


def _sse_blocks_from_bytes(data: bytes) -> list[bytes]:
    if not data:
        return []
    text = data.decode("utf-8")
    normalized = text.replace("\r\n", "\n")
    blocks = normalized.split("\n\n")
    return [block.encode("utf-8") for block in blocks if block.strip()]


def _parse_sse_data_payload(block: bytes) -> tuple[str, Any] | None:
    try:
        text = block.decode("utf-8")
    except UnicodeDecodeError:
        return None
    data_lines = [ln for ln in text.split("\n") if ln.startswith("data:")]
    if not data_lines:
        return None
    payload = "\n".join(ln[len("data:") :].lstrip() for ln in data_lines)
    if payload.strip() == "[DONE]":
        return ("[DONE]", None)
    try:
        return ("json", json.loads(payload))
    except json.JSONDecodeError:
        return ("raw", payload)


def _accumulate_tool_arguments_from_sse_obj(
    obj: Any,
    buffers: dict[tuple[int, int], str],
) -> None:
    if not isinstance(obj, dict):
        return
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        choice_index = int(choice.get("index", 0))
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                _accumulate_tool_arguments_from_message(message, buffers, choice_index)
            continue
        _accumulate_tool_arguments_from_delta(delta, buffers, choice_index)


def _accumulate_tool_arguments_from_message(
    message: dict[str, Any],
    buffers: dict[tuple[int, int], str],
    choice_index: int,
) -> None:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_index = int(tc.get("index", 0))
            fn = tc.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                key = (choice_index, tc_index)
                buffers[key] = buffers.get(key, "") + fn["arguments"]


def _accumulate_tool_arguments_from_delta(
    delta: dict[str, Any],
    buffers: dict[tuple[int, int], str],
    choice_index: int,
) -> None:
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_index = int(tc.get("index", 0))
            fn = tc.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                key = (choice_index, tc_index)
                buffers[key] = buffers.get(key, "") + fn["arguments"]
    function_call = delta.get("function_call")
    if isinstance(function_call, dict) and isinstance(function_call.get("arguments"), str):
        key = (choice_index, 0)
        buffers[key] = buffers.get(key, "") + function_call["arguments"]


def _template_agent_artifact_ids_in_sse_stream(
    data: bytes,
    output_cap_paths: list[str] | None = None,
    write_result_paths: list[str] | None = None,
    view_image_paths: list[str] | None = None,
) -> bytes:
    if not data:
        return data
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return _template_agent_artifact_ids_in_bytes(data)

    blocks = _sse_blocks_from_bytes(data)
    if not blocks:
        return _template_agent_artifact_ids_in_bytes(data)

    parsed_blocks: list[tuple[bytes, str, Any | None]] = []
    buffers: dict[tuple[int, int], str] = {}
    for block in blocks:
        parsed = _parse_sse_data_payload(block)
        if parsed is None:
            parsed_blocks.append((block, "raw", None))
            continue
        kind, payload = parsed
        if kind == "json" and isinstance(payload, dict):
            _accumulate_tool_arguments_from_sse_obj(payload, buffers)
        parsed_blocks.append((block, kind, payload))

    cap_paths = output_cap_paths or []
    write_paths = write_result_paths or []
    view_paths = view_image_paths or []
    templated_buffers = {
        key: _template_tool_call_arguments_string(args, cap_paths, write_paths, view_paths)
        for key, args in buffers.items()
    }
    if not templated_buffers:
        fallback = _template_agent_artifact_ids_in_bytes(data)
        if cap_paths:
            fallback = _template_output_cap_paths_in_bytes(fallback, cap_paths)
        if write_paths:
            fallback = _template_write_result_paths_in_bytes(fallback, write_paths)
        if view_paths:
            fallback = _template_view_image_paths_in_bytes(fallback, view_paths)
        return fallback

    last_tool_event_index: dict[tuple[int, int], int] = {}
    for idx, (_block, kind, payload) in enumerate(parsed_blocks):
        if kind != "json" or not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_index = int(choice.get("index", 0))
            delta = choice.get("delta")
            if isinstance(delta, dict):
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            tc_index = int(tc.get("index", 0))
                            last_tool_event_index[(choice_index, tc_index)] = idx
                if isinstance(delta.get("function_call"), dict):
                    last_tool_event_index[(choice_index, 0)] = idx

    out_parts: list[str] = []
    for idx, (_block, kind, payload) in enumerate(parsed_blocks):
        if kind == "[DONE]":
            out_parts.append("data: [DONE]\n\n")
            continue
        if kind == "raw":
            out_parts.append(_block.decode("utf-8") + "\n\n")
            continue
        if kind != "json" or not isinstance(payload, dict):
            out_parts.append(_block.decode("utf-8") + "\n\n")
            continue

        rebuilt = copy.deepcopy(payload)
        choices = rebuilt.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                choice_index = int(choice.get("index", 0))
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    message = choice.get("message")
                    if isinstance(message, dict):
                        tool_calls = message.get("tool_calls")
                        if isinstance(tool_calls, list):
                            for tc in tool_calls:
                                if not isinstance(tc, dict):
                                    continue
                                tc_index = int(tc.get("index", 0))
                                key = (choice_index, tc_index)
                                if key in templated_buffers and idx == last_tool_event_index.get(
                                    key, idx
                                ):
                                    fn = tc.setdefault("function", {})
                                    if isinstance(fn, dict):
                                        fn["arguments"] = templated_buffers[key]
                    continue
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        tc_index = int(tc.get("index", 0))
                        key = (choice_index, tc_index)
                        fn = tc.setdefault("function", {})
                        if not isinstance(fn, dict):
                            continue
                        if key in templated_buffers and idx == last_tool_event_index.get(key, -1):
                            fn["arguments"] = templated_buffers[key]
                        elif isinstance(fn.get("arguments"), str):
                            fn["arguments"] = ""
                function_call = delta.get("function_call")
                if isinstance(function_call, dict):
                    key = (choice_index, 0)
                    if key in templated_buffers and idx == last_tool_event_index.get(key, -1):
                        function_call["arguments"] = templated_buffers[key]
                    elif isinstance(function_call.get("arguments"), str):
                        function_call["arguments"] = ""

        out_parts.append("data: " + _json_dumps(rebuilt) + "\n\n")

    result = "".join(out_parts).encode("utf-8")
    if cap_paths:
        result = _template_output_cap_paths_in_bytes(result, cap_paths)
    if write_paths:
        result = _template_write_result_paths_in_bytes(result, write_paths)
    if view_paths:
        result = _template_view_image_paths_in_bytes(result, view_paths)
    return result


def _template_response_bytes_for_cache(
    data: bytes,
    output_cap_paths: list[str] | None = None,
    write_result_paths: list[str] | None = None,
    view_image_paths: list[str] | None = None,
) -> bytes:
    cap_paths = output_cap_paths or []
    write_paths = write_result_paths or []
    view_paths = view_image_paths or []
    if not data:
        return data
    if b"data:" in data and (b"tool_calls" in data or b"function_call" in data):
        templated = _template_agent_artifact_ids_in_sse_stream(
            data, cap_paths, write_paths, view_paths
        )
        if (
            AGENT_ARTIFACT_PLACEHOLDER.encode("utf-8") in templated
            or templated != data
            or (
                cap_paths
                and any(_output_cap_path_placeholder(i).encode() in templated for i in range(len(cap_paths)))
            )
            or (
                write_paths
                and any(
                    _write_result_path_placeholder(i).encode() in templated for i in range(len(write_paths))
                )
            )
            or (
                view_paths
                and any(
                    _view_image_path_placeholder(i).encode() in templated for i in range(len(view_paths))
                )
            )
        ):
            return templated
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        out = _template_agent_artifact_ids_in_bytes(data)
        if cap_paths:
            out = _template_output_cap_paths_in_bytes(out, cap_paths)
        if write_paths:
            out = _template_write_result_paths_in_bytes(out, write_paths)
        if view_paths:
            out = _template_view_image_paths_in_bytes(out, view_paths)
        return out
    if isinstance(obj, dict) and "choices" in obj:
        templated_obj = _template_tool_calls_in_json_value(
            obj, cap_paths, write_paths, view_paths
        )
        if settings.cache_template_artifact_ids:
            templated_obj = _template_agent_artifact_ids_in_value(templated_obj)
        if cap_paths:
            templated_obj = _template_output_cap_paths_in_value(templated_obj, cap_paths)
        if write_paths:
            templated_obj = _template_write_result_paths_in_value(templated_obj, write_paths)
        if view_paths:
            templated_obj = _template_view_image_paths_in_value(templated_obj, view_paths)
        return _json_dumps(templated_obj).encode("utf-8")
    out = _template_agent_artifact_ids_in_bytes(data)
    if cap_paths:
        out = _template_output_cap_paths_in_bytes(out, cap_paths)
    if write_paths:
        out = _template_write_result_paths_in_bytes(out, write_paths)
    if view_paths:
        out = _template_view_image_paths_in_bytes(out, view_paths)
    return out


def _substitute_agent_artifact_placeholder_in_bytes(data: bytes, artifact_id: str | None) -> bytes:
    if not artifact_id or not data:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return text.replace(AGENT_ARTIFACT_PLACEHOLDER, artifact_id).encode("utf-8")


def _substitute_cached_response_bytes(
    data: bytes,
    *,
    artifact_id: str | None,
    output_cap_paths: list[str],
    write_result_paths: list[str],
    view_image_paths: list[str],
) -> bytes | None:
    if not data:
        return data
    body = data
    if settings.cache_template_artifact_ids and artifact_id:
        body = _substitute_agent_artifact_placeholder_in_bytes(body, artifact_id)
    if settings.cache_template_output_cap_paths:
        substituted = _substitute_output_cap_placeholders_in_bytes(body, output_cap_paths)
        if substituted is None:
            return None
        body = substituted
    if settings.cache_template_write_result_paths:
        substituted = _substitute_write_result_placeholders_in_bytes(body, write_result_paths)
        if substituted is None:
            return None
        body = substituted
    if settings.cache_template_view_image_paths:
        substituted = _substitute_view_image_placeholders_in_bytes(body, view_image_paths)
        if substituted is None:
            return None
        body = substituted
    return body


def _normalize_body_for_cache_key(body: Any) -> Any:
    body = _normalize_body_for_cache_key_messages(body)
    if settings.cache_template_artifact_ids:
        body = _template_agent_artifact_ids_in_value(body)
    body = _normalize_dynamic_request_fields_in_value(body)
    if settings.cache_template_output_cap_paths:
        cap_paths = _extract_output_cap_paths_from_json(body)
        body = _template_output_cap_paths_in_value(body, cap_paths)
    if settings.cache_template_write_result_paths:
        write_paths = _extract_write_result_paths_from_json(body)
        body = _template_write_result_paths_in_value(body, write_paths)
    if settings.cache_template_view_image_paths:
        view_paths = _extract_view_image_paths_from_json(body)
        body = _template_view_image_paths_in_value(body, view_paths)
    return body


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
        "cache_template_artifact_ids": settings.cache_template_artifact_ids,
        "cache_template_output_cap_paths": settings.cache_template_output_cap_paths,
        "cache_template_write_result_paths": settings.cache_template_write_result_paths,
        "cache_template_view_image_paths": settings.cache_template_view_image_paths,
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


async def _replay(
    record: dict[str, Any],
    *,
    cache_marker: str | None = None,
    artifact_id: str | None = None,
    output_cap_paths: list[str] | None = None,
    write_result_paths: list[str] | None = None,
    view_image_paths: list[str] | None = None,
) -> Response | None:
    status_code = int(record.get("upstream", {}).get("status_code", 200))
    headers = _outbound_headers(record, cache_marker)
    cap_paths = output_cap_paths if output_cap_paths is not None else []
    write_paths = write_result_paths if write_result_paths is not None else []
    view_paths = view_image_paths if view_image_paths is not None else []
    templating_enabled = (
        settings.cache_template_artifact_ids
        or settings.cache_template_output_cap_paths
        or settings.cache_template_write_result_paths
        or settings.cache_template_view_image_paths
    )

    if record.get("stream"):
        chunks = [_from_b64(item) for item in record.get("chunks_b64", [])]
        if templating_enabled:
            payload = _substitute_cached_response_bytes(
                b"".join(chunks),
                artifact_id=artifact_id,
                output_cap_paths=cap_paths,
                write_result_paths=write_paths,
                view_image_paths=view_paths,
            )
            if payload is None:
                return None

            async def generate():
                yield payload
                delay = _replay_delay_s(payload)
                if delay:
                    await asyncio.sleep(delay)

            return StreamingResponse(generate(), status_code=status_code, headers=headers)

        async def generate():
            for chunk in chunks:
                yield chunk
                delay = _replay_delay_s(chunk)
                if delay:
                    await asyncio.sleep(delay)

        return StreamingResponse(generate(), status_code=status_code, headers=headers)

    body = _from_b64(record.get("body_b64", ""))
    if templating_enabled:
        substituted = _substitute_cached_response_bytes(
            body,
            artifact_id=artifact_id,
            output_cap_paths=cap_paths,
            write_result_paths=write_paths,
            view_image_paths=view_paths,
        )
        if substituted is None:
            return None
        body = substituted
    return Response(
        content=body,
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
                "body_b64": _body_b64(
                    _template_response_bytes_for_cache(
                        upstream.content,
                        _extract_output_cap_paths_from_bytes(body)
                        if settings.cache_template_output_cap_paths
                        else None,
                        _extract_write_result_paths_from_bytes(body)
                        if settings.cache_template_write_result_paths
                        else None,
                        _extract_view_image_paths_from_bytes(body)
                        if settings.cache_template_view_image_paths
                        else None,
                    )
                    if (
                        settings.cache_template_artifact_ids
                        or settings.cache_template_output_cap_paths
                        or settings.cache_template_write_result_paths
                        or settings.cache_template_view_image_paths
                    )
                    else upstream.content
                ),
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
                stream_body = b"".join(chunks)
                if (
                    settings.cache_template_artifact_ids
                    or settings.cache_template_output_cap_paths
                    or settings.cache_template_write_result_paths
                    or settings.cache_template_view_image_paths
                ):
                    stream_body = _template_response_bytes_for_cache(
                        stream_body,
                        _extract_output_cap_paths_from_bytes(body)
                        if settings.cache_template_output_cap_paths
                        else None,
                        _extract_write_result_paths_from_bytes(body)
                        if settings.cache_template_write_result_paths
                        else None,
                        _extract_view_image_paths_from_bytes(body)
                        if settings.cache_template_view_image_paths
                        else None,
                    )
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
                        "chunks_b64": [_body_b64(stream_body)],
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
    artifact_id = (
        _extract_agent_artifact_id_from_bytes(body) if settings.cache_template_artifact_ids else None
    )
    output_cap_paths = (
        _extract_output_cap_paths_from_bytes(body)
        if settings.cache_template_output_cap_paths
        else []
    )
    write_result_paths = (
        _extract_write_result_paths_from_bytes(body)
        if settings.cache_template_write_result_paths
        else []
    )
    view_image_paths = (
        _extract_view_image_paths_from_bytes(body)
        if settings.cache_template_view_image_paths
        else []
    )
    cached = _read_record(key)
    if cached is not None:
        replayed = await _replay(
            cached,
            cache_marker="hit",
            artifact_id=artifact_id,
            output_cap_paths=output_cap_paths,
            write_result_paths=write_result_paths,
            view_image_paths=view_image_paths,
        )
        if replayed is not None:
            return replayed

    lock = await _lock_for(key)
    await lock.acquire()
    stream_request = _is_stream_request(body)
    try:
        cached = _read_record(key)
        if cached is not None:
            lock.release()
            replayed = await _replay(
                cached,
                cache_marker="hit",
                artifact_id=artifact_id,
                output_cap_paths=output_cap_paths,
                write_result_paths=write_result_paths,
                view_image_paths=view_image_paths,
            )
            if replayed is not None:
                return replayed

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
