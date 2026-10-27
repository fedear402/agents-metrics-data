#!/usr/bin/env bash
# ============================================================================
# Container entrypoint for the R3 full-agent condition.
#
# Sequence:
#   1. (root) run the egress firewall, if NET_ADMIN is available.
#   2. (root) make /work writable by the unprivileged `bench` user.
#   3. drop privileges to `bench` and exec the passed command (normally
#      `claude -p ...`), optionally under a wall-clock `timeout`.
#
# The agentic prompt is delivered on STDIN by the host wrapper, so we exec the
# command directly and let it inherit stdin/stdout/stderr.
# ============================================================================
set -euo pipefail

run_as_root() { [ "$(id -u)" -eq 0 ]; }

# --- 1. egress firewall (best-effort; requires --cap-add=NET_ADMIN/NET_RAW) ---
if run_as_root; then
    if iptables -L >/dev/null 2>&1; then
        /usr/local/bin/init-firewall.sh
    else
        # No NET_ADMIN -> cannot lock the network. Refuse unless explicitly
        # allowed, so a misconfigured launch never runs bypassPermissions with
        # an open network.
        if [ "${ALLOW_OPEN_NETWORK:-0}" != "1" ]; then
            echo "[entrypoint] FATAL: no NET_ADMIN (cannot apply egress firewall)." >&2
            echo "[entrypoint] Re-run with --cap-add=NET_ADMIN --cap-add=NET_RAW," >&2
            echo "[entrypoint] or set ALLOW_OPEN_NETWORK=1 to override (NOT recommended)." >&2
            exit 78
        fi
        echo "[entrypoint] WARNING: running with OPEN network (ALLOW_OPEN_NETWORK=1)." >&2
    fi
fi

# --- 2. ensure the bind-mounted workdir is writable by bench ---
if run_as_root; then
    chown -R bench:bench /work 2>/dev/null || true
fi

# --- 3. drop privileges and exec the command (under optional timeout) ---
# AGENTIC_TIMEOUT (seconds) is an in-container wall-clock backstop; the host
# wrapper enforces its own hard kill as well.
CMD=("$@")
if [ "${#CMD[@]}" -eq 0 ]; then
    CMD=(claude --version)
fi

if [ -n "${AGENTIC_TIMEOUT:-}" ]; then
    CMD=(timeout --signal=TERM --kill-after=30 "${AGENTIC_TIMEOUT}" "${CMD[@]}")
fi

if run_as_root; then
    exec gosu bench:bench "${CMD[@]}"
else
    exec "${CMD[@]}"
fi
