#!/bin/sh
# Start Claude Code 2.1.278 for the Haiku 5.5 API-key collection
# (quality_reports/decision_records/2026-10-16_haiku55-api-collection.md).
# The key comes from the file named by HAIKU_API_KEY_FILE and goes only to this process.
# CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1 keeps it out of the agent's shell commands.
# DISABLE_AUTOUPDATER=1 keeps this older binary from updating the installed Claude Code.
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
bin="$root/data/collections/haiku55_api_bin/claude-2.1.278"
if [ -z "${HAIKU_API_KEY_FILE:-}" ] || [ ! -r "$HAIKU_API_KEY_FILE" ]; then
    echo "haiku_api_claude.sh: HAIKU_API_KEY_FILE does not name a readable file" >&2
    exit 64
fi
ANTHROPIC_API_KEY=$(tr -d '\r\n' < "$HAIKU_API_KEY_FILE")
export ANTHROPIC_API_KEY
export CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1
export DISABLE_AUTOUPDATER=1
unset HAIKU_API_KEY_FILE
exec "$bin" "$@"
