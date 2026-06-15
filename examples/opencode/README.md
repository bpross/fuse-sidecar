# Using fuse-sidecar with opencode

This directory contains a working opencode configuration that routes the
`fuse` agent through your local `fuse-sidecar` server.

## How it works

The sidecar exposes an OpenAI-compatible `/v1/chat/completions` endpoint at
`http://127.0.0.1:7777`. opencode talks to it via the bundled
`@ai-sdk/openai-compatible` provider — no plugin code required.

When you select the `fuse` agent (Tab through your primary agents in the
TUI), every chat completion request goes to the sidecar. The sidecar then
runs the speculative → panel → judge → primary pipeline described in the
top-level README.

## Setup

### 1. Start `fuse-sidecar`

Make sure the sidecar is running and healthy:

```sh
# from the top of the fuse-sidecar repo
go build -o fuse-sidecar ./cmd/fuse-sidecar

mkdir -p ~/.config/fuse-sidecar
cp config.example.json ~/.config/fuse-sidecar/config.json
# edit the config to taste

export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

./fuse-sidecar &

curl -s http://127.0.0.1:7777/healthz
# {"ok":true,"version":"dev"}
```

### 2. Drop the config into opencode

Either copy the config wholesale:

```sh
cp examples/opencode/opencode.json ~/.config/opencode/opencode.json
```

Or merge the `provider.fuse` and `agent.fuse` blocks into your existing
`opencode.json`. The block under `provider.fuse.models` must list every
sidecar model ID you want opencode to see (e.g. `fusion-plan`,
`fusion-code`).

### 3. Use it

Launch opencode, then press **Tab** until the active agent shows `fuse`.
From there, ask a planning question. opencode will:

- send the speculative call through your sidecar
- pass through any tool-call turns (read/grep/glob/lsp/etc.) at native speed
- run the full panel + judge + primary fusion on the final answer turn

You'll see `Thought:` style reasoning blocks in the TUI as the sidecar
emits `fuse: evaluating turn` / `querying panel` / `judging` / `writing
final answer` progress events — opencode renders them inline.

## Verifying the wiring

After a session, inspect what actually happened:

```sh
# Recent decisions
ls -t ~/.local/share/fuse-sidecar/runs/ | head -5 | while read f; do
  jq '{decision, latency_ms: .total_latency_ms, final_bytes: .final_answer_bytes,
       panel: [.panel[]? | {model, latency_ms, ok, attempts}]}' \
    ~/.local/share/fuse-sidecar/runs/$f
done

# Or hit the admin endpoint
curl -s http://127.0.0.1:7777/admin/status | jq '.recent[0:5]'
```

A healthy planning session looks like:

- multiple `decision: "passthrough"` turns (tool-call investigation)
- one `decision: "fusion"` turn at the end (the final plan) with all panel
  members `ok: true, attempts: 1`

## Switching to fusion-code

The bundled config defaults the `fuse` agent to `fuse/fusion-plan`. To use
`fusion-code` instead (different model triple — see
`config.example.json`), change the agent's `model` field:

```jsonc
{
  "agent": {
    "fuse": {
      "model": "fuse/fusion-code"
    }
  }
}
```

Or add a second agent so you can Tab between them:

```jsonc
{
  "agent": {
    "fuse-plan": {
      "mode": "primary",
      "model": "fuse/fusion-plan",
      "description": "Plan with fused panel deliberation",
      "permission": { "edit": "deny", "bash": "deny" }
    },
    "fuse-code": {
      "mode": "primary",
      "model": "fuse/fusion-code",
      "description": "Code with fused panel deliberation"
    }
  }
}
```

## Permissions

The sample config sets `edit: deny` and `bash: deny` on the `fuse` agent
because plan-style deliberation shouldn't be the thing that writes to your
filesystem. Drop those (or set them to `ask`) if you want the fuse agent
to make changes directly.

## Troubleshooting

**"model_not_found" from opencode** — the model IDs under
`provider.fuse.models` must match the IDs your sidecar config defines
under `models`. Run `curl http://127.0.0.1:7777/v1/models` to see what
the sidecar exposes.

**Long pauses with no output** — the sidecar emits SSE heartbeats (`:
ping`) every 5s during the panel/judge gap. If opencode shows nothing for
30+ seconds and the snapshot eventually records a fusion success, that's
normal: the panel is just slow. Check `panel_timeout_ms` in your sidecar
config if you want a tighter cap.

**Panel members failing** — `~/.local/share/fuse-sidecar/runs/<latest>.json`
records the per-member error string. Common causes:

- `temperature is deprecated for this model` → remove the `temperature`
  override on that endpoint in the sidecar config
- `Unsupported parameter: 'max_tokens'` → the OpenAI provider auto-routes
  to `max_completion_tokens` for gpt-5/o-series; if you see this on a
  newer model, the provider list in `internal/providers/openai.go` may
  need an update
- `response had empty content (finish="tool_calls", ...)` → should no
  longer happen as of the tools-stripping fix; if you see it, file an
  issue

**Sidecar isn't picking up config changes** — `kill -HUP $(pgrep
fuse-sidecar)` triggers a hot reload without dropping in-flight requests.
