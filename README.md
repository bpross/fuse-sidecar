# fuse-sidecar

A local OpenAI-compatible HTTP server that runs an OpenRouter-style fusion
pipeline: a panel of N models answers in parallel, a judge produces structured
comparison, and the primary model writes the final answer using that analysis.
Any client that speaks `POST /v1/chat/completions` (opencode, Claude Code,
curl, etc.) can point at it.

Inspired by [OpenRouter's Fusion plugin](https://openrouter.ai/docs/guides/features/plugins/fusion).

## How it works

For each `POST /v1/chat/completions`:

1. **Speculative primary call** runs the original request against the
   configured primary model with all tools attached. Response is buffered, not
   streamed to the client yet.
2. **Detect.** If the response contains any tool calls, it is a tool-call turn
   — the buffered response is re-emitted as SSE and the request is done.
   No fusion overhead on tool-call turns.
3. **Fuse.** If it is a text-only finalization turn:
   1. Fan out the original request to a panel of N models in parallel
      (configurable wall-clock cap, ≥K successes required).
   2. The judge model receives all panel responses and returns structured JSON
      analysis — consensus, contradictions, partial coverage, unique insights,
      blind spots. The judge **compares**, it does not merge.
   3. The primary model is called again with the original conversation plus a
      synthetic assistant+user turn carrying the judge analysis. Its streaming
      response is forwarded directly to the client.
4. **Fallback.** Any failure along the fusion path falls back to streaming the
   buffered speculative response. The client always gets a valid answer or a
   clean error — never a hung stream.

## Providers

| Provider   | ID          | Notes |
|------------|-------------|-------|
| Anthropic  | `anthropic` | Messages API, streaming + buffered + tool calls |
| OpenAI     | `openai`    | Chat Completions API |
| _anything_ | _custom_    | Any other ID with a `base_url` is treated as a generic OpenAI-compatible endpoint (OpenRouter, LM Studio, llama.cpp, Together, Groq, etc.) |

## Build and run

```sh
git clone https://github.com/bpross/fuse-sidecar.git
cd fuse-sidecar
go build -o fuse-sidecar ./cmd/fuse-sidecar

mkdir -p ~/.config/fuse-sidecar
cp config.example.json ~/.config/fuse-sidecar/config.json
# edit the config; only the providers and models you intend to use need keys

export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

./fuse-sidecar --config ~/.config/fuse-sidecar/config.json
```

Or with `make`:

```sh
make build
ANTHROPIC_API_KEY=sk-ant-... CONFIG=~/.config/fuse-sidecar/config.json make run
```

The server defaults to `127.0.0.1:7777`.

## Verify it's running

```sh
curl -s http://127.0.0.1:7777/healthz
# {"ok":true,"version":"dev"}

curl -s http://127.0.0.1:7777/v1/models | jq .
# { "object": "list", "data": [ { "id": "fusion-plan", ... } ] }
```

Send a streaming completion:

```sh
curl -s -N http://127.0.0.1:7777/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "fusion-plan",
    "stream": true,
    "messages": [
      {"role": "user", "content": "What are the strongest arguments for and against carbon taxes?"}
    ]
  }'
```

## Endpoints

| Method | Path                       | Purpose |
|--------|----------------------------|---------|
| POST   | `/v1/chat/completions`     | OpenAI-compatible chat completions (streaming only) |
| GET    | `/v1/models`               | List configured logical model IDs |
| GET    | `/healthz`                 | Liveness check |
| GET    | `/admin/status`            | Last 50 request decisions for debugging |
| GET    | `/metrics`                 | Prometheus-style text metrics |

## Configuration

See `config.example.json` for a full annotated example. Key fields:

- `listen` — bind address (default `127.0.0.1:7777`).
- `log_dir` — slog output and per-request snapshot files (tilde-expanded).
- `reasoning_blocks_enabled` — emit `reasoning_content` SSE deltas during
  fusion progress. Clients that don't render reasoning will ignore them
  harmlessly; clients that do (e.g. opencode) will show fusion status.
- `snapshot_retention` — max number of per-request JSON snapshot files to keep.
- `providers.<id>` — credential mapping. `api_key_env` is the name of the env
  var that holds the key. `base_url` overrides the default endpoint. `headers`
  are extra HTTP headers attached to every outgoing request.
- `models.<id>` — one logical model name (e.g. `fusion-plan`) and the
  primary/panel/judge endpoints behind it. Clients send `model: "<id>"` in the
  request body.
  - `panel_timeout_ms` — wall-clock cap for the panel fan-out (default 25000).
  - `panel_min_success` — minimum successful panel responses required to
    proceed to the judge (default 2). Below this, falls back to speculative.

Validation runs at startup and fails fast: unknown providers, unset API key
envs, panels with fewer than 1 entry, and `panel_min_success` exceeding panel
size all abort startup with a concrete error.

## Hot reload

`kill -HUP <pid>` re-reads the config file and atomically swaps the provider
registry and model definitions. In-flight requests continue against the old
state until they complete; new requests see the new state. Failed reloads
keep the old config running and log the error.

## Snapshots

Each fusion or fallback decision writes a JSON snapshot to
`<log_dir>/runs/<timestamp>-<request_id>.json` containing:

- The decision (passthrough, fusion, or fallback) and the fallback reason
- Per-panel-member provider, model, latency, success/error
- Judge analysis JSON
- Total latency in ms

These are pruned to `snapshot_retention` by lexicographic order.

## Hooking it up to clients

This repo only ships the server. Client plugins live elsewhere (coming
soon). For now, configure clients manually.

### opencode

A working configuration and step-by-step guide live in
[`examples/opencode/`](./examples/opencode/). The short version:

```sh
cp examples/opencode/opencode.json ~/.config/opencode/opencode.json
```

Then launch opencode, Tab to the `fuse` agent, and your requests route
through the sidecar. The example covers both `fusion-plan` and
`fusion-code` model IDs, reasoning-block progress rendering in the TUI,
and per-snapshot debugging.

### Claude Code

Coming soon. The sidecar exposes the same OpenAI-compatible surface that
Claude Code's custom-provider support targets.

## Development

```sh
make test    # go test ./...
make vet     # go vet ./...
make check   # vet + test
```

Repo layout:

```
fuse-sidecar/
├── cmd/fuse-sidecar/         # entrypoint, signal handling, reload loop
├── internal/
│   ├── config/               # schema, load, validate
│   ├── fusion/               # speculative → detect → panel → judge → handoff
│   ├── obs/                  # slog, snapshot writer, metrics, status ring
│   ├── providers/            # Provider interface; Anthropic, OpenAI
│   └── server/               # HTTP, SSE encoder, request handlers
├── config.example.json
└── README.md
```

## License

MIT. See `LICENSE`.
