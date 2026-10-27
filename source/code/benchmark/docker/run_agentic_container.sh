#!/usr/bin/env bash
# ============================================================================
# run_agentic_container.sh: launch ONE ephemeral R3 full-agent run in Docker.
#
# Per the 2026-06-25 R3 build plan, this is the production-isolation
# path: a fresh container per run that mounts ONLY the per-run working directory
# (seeded with input.csv) — NOT the repo, NOT data/tasks/references — runs `claude -p`
# with bypassPermissions inside it, with egress restricted to the Anthropic API,
# and leaves result.json + the agent's script in the workdir for copy-back.
#
# It is a drop-in replacement for the host-local `run_claude_agentic()` call in
# harness.py: same contract (workdir in, result.json + run.* out, stream-json on
# stdout), stronger isolation. Run it UNDER the harness PacingController (one
# paced run = one container, sequential) so subscription headroom is preserved —
# the container adds no concurrency of its own.
#
# Usage:
#   run_agentic_container.sh --workdir DIR [options]
#
# Required:
#   --workdir DIR        Host per-run dir; MUST already contain input.csv.
#                        This is the ONLY thing mounted into the container.
#
# Options:
#   --prompt-file FILE   File with the agentic prompt (default: read stdin).
#   --image NAME         Image tag (default: agents-metrics-agentic:latest).
#   --model NAME         Model passed to claude (default: sonnet).
#   --max-turns N        Circuit-breaker turn cap (default: 60).
#   --timeout SECONDS    Wall-clock cap (default: 1800). Enforced both inside
#                        the container and by a host watchdog.
#   --auth MODE          oauth | apikey (default: oauth). See README "Auth".
#                          oauth  -> subscription token via CLAUDE_CODE_OAUTH_TOKEN
#                                    (mint once with `claude setup-token`).
#                          apikey -> ANTHROPIC_API_KEY (bills API credits, NOT
#                                    the subscription — breaks pacing accounting).
#   --memory SIZE        Container memory cap (default: 4g).
#   --cpus N             Container CPU cap (default: 2).
#   --open-network       DEBUG ONLY: skip the egress firewall.
#
# Outputs (left in --workdir):
#   result.json              the agent's answer (if produced)
#   claude_stream.jsonl      full stream-json stdout (cost / tool-call telemetry)
#   claude_stderr.txt        container stderr
#   the agent's script(s)    (normalize to run.{py,R,do} via harness.copy_back)
#
# Exit code: the container's exit code (124 = timed out).
# ============================================================================
set -euo pipefail

# ---- defaults ----
WORKDIR=""
PROMPT_FILE=""
IMAGE="agents-metrics-agentic:latest"
MODEL="sonnet"
MAX_TURNS="60"
TIMEOUT="1800"
AUTH="oauth"
MEMORY="4g"
CPUS="2"
OPEN_NETWORK="0"

die() { echo "run_agentic_container: $*" >&2; exit 2; }

# ---- parse args ----
while [ "$#" -gt 0 ]; do
    case "$1" in
        --workdir)      WORKDIR="$2"; shift 2;;
        --prompt-file)  PROMPT_FILE="$2"; shift 2;;
        --image)        IMAGE="$2"; shift 2;;
        --model)        MODEL="$2"; shift 2;;
        --max-turns)    MAX_TURNS="$2"; shift 2;;
        --timeout)      TIMEOUT="$2"; shift 2;;
        --auth)         AUTH="$2"; shift 2;;
        --memory)       MEMORY="$2"; shift 2;;
        --cpus)         CPUS="$2"; shift 2;;
        --open-network) OPEN_NETWORK="1"; shift;;
        -h|--help)      sed -n '2,60p' "$0"; exit 0;;
        *)              die "unknown argument: $1";;
    esac
done

command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
[ -n "$WORKDIR" ] || die "--workdir is required"
[ -d "$WORKDIR" ] || die "workdir does not exist: $WORKDIR"
[ -f "$WORKDIR/input.csv" ] || die "workdir missing input.csv: $WORKDIR/input.csv"

# Absolute path for the bind mount.
WORKDIR="$(cd "$WORKDIR" && pwd)"

# ---- auth wiring ----
AUTH_ARGS=()
case "$AUTH" in
    oauth)
        [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] || die \
            "auth=oauth needs CLAUDE_CODE_OAUTH_TOKEN in the environment.
   Mint one on the host with:  claude setup-token
   then:  export CLAUDE_CODE_OAUTH_TOKEN=...   (subscription billing; pacing-consistent)"
        AUTH_ARGS+=(-e "CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}")
        ;;
    apikey)
        [ -n "${ANTHROPIC_API_KEY:-}" ] || die \
            "auth=apikey needs ANTHROPIC_API_KEY in the environment.
   NOTE: this bills API credits, NOT the subscription — it bypasses the
   pacing controller's subscription accounting."
        AUTH_ARGS+=(-e "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
        ;;
    *)
        die "unknown --auth mode: $AUTH (use oauth|apikey)"
        ;;
esac

# ---- network isolation ----
NET_ARGS=()
if [ "$OPEN_NETWORK" = "1" ]; then
    echo "run_agentic_container: WARNING running with OPEN network (debug)." >&2
    NET_ARGS+=(-e "ALLOW_OPEN_NETWORK=1")
else
    # NET_ADMIN/NET_RAW let entrypoint.sh apply the egress firewall that pins
    # outbound traffic to the Anthropic API endpoint only.
    NET_ARGS+=(--cap-add=NET_ADMIN --cap-add=NET_RAW)
fi

NAME="agentic-$(date +%s)-$$-$RANDOM"

# ---- docker run argv ----
# Hardening: drop all caps except the two the firewall needs, no privilege
# escalation, capped memory/cpu/pids, ephemeral (--rm), and the ONLY bind mount
# is the per-run workdir. The repo and data/tasks/references are never visible.
DOCKER_ARGS=(
    run --rm -i
    --name "$NAME"
    --cap-drop=ALL
    --security-opt no-new-privileges
    --memory "$MEMORY" --memory-swap "$MEMORY"
    --cpus "$CPUS"
    --pids-limit 512
    --tmpfs /tmp:rw,nosuid,nodev,size=512m
    -v "$WORKDIR":/work
    -w /work
    -e "AGENTIC_TIMEOUT=${TIMEOUT}"
)
DOCKER_ARGS+=("${NET_ARGS[@]}")
DOCKER_ARGS+=("${AUTH_ARGS[@]}")
DOCKER_ARGS+=("$IMAGE")

# Command run inside the container (entrypoint drops to `bench` then execs this).
# Matches harness.agentic_claude_cmd(): full default tool surface (no
# allow/disallow flags), bypassPermissions, stream-json telemetry.
CLAUDE_CMD=(
    claude -p
    --model "$MODEL"
    --output-format stream-json --verbose --include-partial-messages
    --max-turns "$MAX_TURNS"
    --no-session-persistence
    --permission-mode bypassPermissions
    --add-dir /work
)
DOCKER_ARGS+=("${CLAUDE_CMD[@]}")

# ---- host watchdog: hard-kill the container if it overruns ----
# (Independent of the in-container `timeout`; covers a wedged docker process.)
GRACE=60
( sleep "$((TIMEOUT + GRACE))"; docker kill "$NAME" >/dev/null 2>&1 || true ) &
WATCHDOG=$!
cleanup() { kill "$WATCHDOG" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# ---- launch ----
# Prompt on stdin -> claude -p; stream-json -> claude_stream.jsonl; stderr file.
STREAM_OUT="$WORKDIR/claude_stream.jsonl"
STDERR_OUT="$WORKDIR/claude_stderr.txt"

set +e
if [ -n "$PROMPT_FILE" ]; then
    [ -f "$PROMPT_FILE" ] || die "prompt file not found: $PROMPT_FILE"
    docker "${DOCKER_ARGS[@]}" < "$PROMPT_FILE" >"$STREAM_OUT" 2>"$STDERR_OUT"
else
    docker "${DOCKER_ARGS[@]}" >"$STREAM_OUT" 2>"$STDERR_OUT"
fi
RC=$?
set -e

if [ -f "$WORKDIR/result.json" ]; then
    echo "run_agentic_container: ok rc=$RC, result.json produced in $WORKDIR" >&2
else
    echo "run_agentic_container: rc=$RC, NO result.json (see $STDERR_OUT)" >&2
fi

exit "$RC"
