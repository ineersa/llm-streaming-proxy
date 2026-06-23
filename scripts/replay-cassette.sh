#!/usr/bin/env bash
# Replay a llama-proxy cassette against the local proxy and print timing + cache headers.
set -euo pipefail

CACHE_DIR="${LLAMA_PROXY_CACHE_DIR:-/var/cache/llama-proxy}"
BASE_URL="${LLAMA_PROXY_REPLAY_URL:-http://127.0.0.1:9052}"

BODY_FILE=""
HEADERS_FILE=""
OUT_FILE=""

cleanup() {
  rm -f -- "${BODY_FILE:-}" "${HEADERS_FILE:-}" "${OUT_FILE:-}"
}
trap cleanup EXIT

usage() {
  cat <<'EOF'
Usage: replay-cassette.sh [CASSETTE.json]

Re-POST a recorded cassette to the llama-proxy and print wall-clock time_total
and response header summary (including x-llama-proxy-cache on cache hits).

If CASSETTE is omitted, uses the smallest *.json file under the cache dir
(default: /var/cache/llama-proxy).

Environment:
  LLAMA_PROXY_CACHE_DIR   cache root (default: /var/cache/llama-proxy)
  LLAMA_PROXY_REPLAY_URL  proxy base URL (default: http://127.0.0.1:9052)

Requires: curl, jq (sudo jq used when cassette is root-owned).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

jq_read() {
  local file="$1"
  local filter="$2"
  if jq -r "$filter" "$file" 2>/dev/null; then
    return 0
  fi
  sudo jq -r "$filter" "$file"
}

jq_compact() {
  local file="$1"
  local filter="$2"
  local out="$3"
  if jq -c "$filter" "$file" >"$out" 2>/dev/null; then
    return 0
  fi
  sudo jq -c "$filter" "$file" >"$out"
}

pick_smallest_cassette() {
  local smallest=""
  local size path

  if [[ ! -d "$CACHE_DIR" ]]; then
    echo "error: cache dir not found: $CACHE_DIR" >&2
    exit 1
  fi

  if smallest=$(find "$CACHE_DIR" -type f -name '*.json' -printf '%s %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2-); then
    :
  elif smallest=$(sudo find "$CACHE_DIR" -type f -name '*.json' -printf '%s %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2-); then
    :
  else
    echo "error: no cassette JSON files under $CACHE_DIR" >&2
    exit 1
  fi

  if [[ -z "$smallest" || ! -f "$smallest" ]]; then
    echo "error: could not pick a cassette from $CACHE_DIR" >&2
    exit 1
  fi
  echo "$smallest"
}

CASSETTE="${1:-$(pick_smallest_cassette)}"
if [[ ! -f "$CASSETTE" ]]; then
  echo "error: cassette not found: $CASSETTE" >&2
  exit 1
fi

for cmd in curl jq; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command not found: $cmd" >&2
    exit 1
  fi
done

BODY_FILE="$(mktemp)"
HEADERS_FILE="$(mktemp)"
OUT_FILE="$(mktemp)"

PATH_JSON="$(jq_read "$CASSETTE" '.request.path')"
STREAM="$(jq_read "$CASSETTE" '.stream')"
jq_compact "$CASSETTE" '.request.body' "$BODY_FILE"

if [[ -z "$PATH_JSON" || "$PATH_JSON" == "null" ]]; then
  echo "error: cassette missing .request.path" >&2
  exit 1
fi

URL="${BASE_URL%/}${PATH_JSON}"

echo "cassette: $CASSETTE"
echo "url:      POST $URL"
echo "stream:   $STREAM"

CURL_OPTS=(-sS -D "$HEADERS_FILE" -o "$OUT_FILE" -w 'time_total=%{time_total}\n' -X POST "$URL" -H 'Content-Type: application/json' --data-binary @"$BODY_FILE")
if [[ "$STREAM" == "true" ]]; then
  CURL_OPTS=(-N "${CURL_OPTS[@]}")
fi

curl "${CURL_OPTS[@]}"

echo "--- headers (summary) ---"
grep -iE '^(HTTP/|content-type:|x-llama-proxy-cache:|transfer-encoding:)' "$HEADERS_FILE" || true
if ! grep -qi '^x-llama-proxy-cache:' "$HEADERS_FILE"; then
  echo "x-llama-proxy-cache: (not present — likely cache miss or pass-through)"
fi