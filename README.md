# llama-proxy

Recording/replay proxy for llama.cpp/OpenAI-compatible test servers.

Intended setup:

- tests keep talking to `:9052`
- real larger llama.cpp server keeps running on `:8052`
- first unique request is proxied to `:8052` and recorded
- later identical requests are replayed from disk, including streaming SSE chunks/tool-call deltas

## Install and run locally

Uses [uv](https://docs.astral.sh/uv/) for the project virtualenv and locked dependencies.

```bash
cd /home/ineersa/projects/llama-proxy
uv sync
LLAMA_PROXY_UPSTREAM=http://127.0.0.1:8052 \
LLAMA_PROXY_PORT=9052 \
LLAMA_PROXY_CACHE_DIR=./llama_proxy_cache \
uv run python -m llama_proxy
```

Or after `uv sync`, without `uv run`:

```bash
.venv/bin/python -m llama_proxy
```

The `llama-proxy` console script is also available: `uv run llama-proxy`.

## Admin endpoints

```bash
curl http://127.0.0.1:9052/__llama_proxy/health
curl http://127.0.0.1:9052/__llama_proxy/cache/stats
curl -X POST http://127.0.0.1:9052/__llama_proxy/cache/clear
```

To time a cache replay from disk (smallest cassette by default, or pass a path):

```bash
./scripts/replay-cassette.sh
./scripts/replay-cassette.sh /var/cache/llama-proxy/3d/<hash>.json
```

Uses `sudo jq` when cassettes under `/var/cache/llama-proxy` are root-owned.

If `LLAMA_PROXY_ADMIN_TOKEN` is set, cache admin endpoints require:

```bash
-H 'X-Llama-Proxy-Token: ...'
```

## Configuration

| env | default | meaning |
| --- | --- | --- |
| `LLAMA_PROXY_UPSTREAM` | `http://127.0.0.1:8052` | real model server |
| `LLAMA_PROXY_HOST` | `0.0.0.0` | bind host |
| `LLAMA_PROXY_PORT` | `9052` | bind port |
| `LLAMA_PROXY_CACHE_DIR` | `./llama_proxy_cache` | cassette storage |
| `LLAMA_PROXY_CACHE_PATHS` | `/completion,/v1/chat/completions,/v1/completions,/v1/responses` | comma-separated POST paths to cache |
| `LLAMA_PROXY_REPLAY_TPS` | `1000` | approximate replay speed for recorded streaming chunks; set `0` for immediate replay |
| `LLAMA_PROXY_REPLAY_CHUNK_DELAY_MS` | `0` | minimum delay after each replayed chunk |
| `LLAMA_PROXY_REQUEST_TIMEOUT_S` | `600` | upstream timeout; `0`/`none` disables |
| `LLAMA_PROXY_ADMIN_TOKEN` | unset | optional admin endpoint token |
| `LLAMA_PROXY_CACHE_NORMALIZE_MESSAGES` | `true` | when enabled, cache key ignores volatile chat prologue (see below); set `false` for full-body keys |

## Cache key

The cache key is SHA-256 over canonical JSON containing:

- HTTP method
- request path
- raw query string
- parsed JSON request body used for the key (object keys sorted)

When `LLAMA_PROXY_CACHE_NORMALIZE_MESSAGES` is enabled (default), bodies with a `messages` array are normalized **only for the key**:

1. Drop leading `system` and `developer` messages.
2. Drop leading `user` messages whose text starts with `[user-context] ` (string content or OpenAI-style text parts).
3. Stop at the first remaining message; that message and everything after it are kept unchanged.

Upstream requests on a cache miss still use the **full original** body. Recorded cassettes store `request.body` (original) and `request.key_body` (normalized key material) for debugging.

Non-JSON bodies and JSON without `messages` are keyed on the full parsed body (or raw body hash) as before.

That means changing model parameters, tools, or `stream` still produces a new key; volatile system prompts and `[user-context]` prologue usually do not, once the tail of the conversation matches a prior recording.

## systemd

A starter unit is in `systemd/llama-proxy.service`. It runs `.venv/bin/python -m llama_proxy` from the repo (create the env with `uv sync` first).

```bash
cd /home/ineersa/projects/llama-proxy
uv sync
sudo mkdir -p /var/cache/llama-proxy
sudo chown "$USER":"$USER" /var/cache/llama-proxy
sudo cp systemd/llama-proxy.service /etc/systemd/system/llama-proxy.service
sudo systemctl daemon-reload
sudo systemctl enable --now llama-proxy.service
```

Use one uvicorn worker. The proxy uses in-process per-key locks so parallel identical cache misses do not stampede upstream.

## Replace the old test server (ops)

After the proxy unit is installed, stop the old llama.cpp test server that was binding `:9052` (for example `test.service` from `.pi/plans/test.service`, or a unit you named `llama-test.service`). The **proxy** unit name depends on what you copied into `/etc/systemd/system/` — commonly `llama-proxy.service` or your own name such as `llama-test.service`.

```bash
# Pick names that match your machine (examples on both lines)
OLD=test.service          # or: llama-test.service, etc.
PROXY=llama-proxy.service # or: llama-test.service if you installed the proxy under that name

sudo systemctl disable --now "$OLD"
sudo systemctl enable --now "$PROXY"
sudo systemctl status "$PROXY"
journalctl -u "$PROXY" -f
```

Confirm the proxy (not `llama-server`) is answering on `:9052`:

```bash
curl http://127.0.0.1:9052/__llama_proxy/health
```

Keep the larger model on `:8052` running; the proxy forwards cache misses there.
