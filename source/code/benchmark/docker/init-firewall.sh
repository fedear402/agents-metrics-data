#!/usr/bin/env bash
# ============================================================================
# Egress allowlist for the R3 agentic container.
# Locks outbound network to: DNS + the Anthropic API endpoint only.
# Everything else (the wider internet, package registries, GitHub, the host
# network) is DROPPED. This is defense-in-depth on top of the "no repo mount"
# isolation: even with bypassPermissions the agent cannot exfiltrate data or
# pull arbitrary code at run time.
#
# Requires the container to be started with: --cap-add=NET_ADMIN --cap-add=NET_RAW
# Runs as root from entrypoint.sh BEFORE privileges are dropped to `bench`.
#
# Modeled on Anthropic's reference devcontainer init-firewall.sh, trimmed to the
# single endpoint this workload needs at run time (no GitHub / npm: the image is
# pre-baked, so nothing is installed during a run).
#
# Set FIREWALL=off in the environment to skip (e.g. for local debugging only).
# ============================================================================
set -euo pipefail

if [ "${FIREWALL:-on}" = "off" ]; then
    echo "[firewall] FIREWALL=off -> skipping egress lockdown (DEBUG ONLY)" >&2
    exit 0
fi

# Hostnames the run is allowed to reach. Keep this list minimal.
# api.anthropic.com is the only endpoint claude -p needs once telemetry,
# error-reporting, and the autoupdater are disabled (see Dockerfile ENV).
ALLOWED_HOSTS=(
    "api.anthropic.com"
)

echo "[firewall] resolving allowed hosts and building ipset..." >&2

# Fresh ipset of allowed destination IPs.
ipset destroy allowed-api 2>/dev/null || true
ipset create allowed-api hash:ip

resolve_into_set() {
    local host="$1"
    # getent uses the container resolver; collect all A records.
    local ips
    ips="$(getent ahostsv4 "$host" | awk '{print $1}' | sort -u || true)"
    if [ -z "$ips" ]; then
        echo "[firewall] FATAL: could not resolve $host" >&2
        exit 1
    fi
    for ip in $ips; do
        ipset add allowed-api "$ip" 2>/dev/null || true
        echo "[firewall]   allow $host -> $ip" >&2
    done
}

for h in "${ALLOWED_HOSTS[@]}"; do
    resolve_into_set "$h"
done

# Identify the DNS server(s) the container uses so we can permit lookups.
DNS_SERVERS="$(awk '/^nameserver/ {print $2}' /etc/resolv.conf | sort -u || true)"

# ---- iptables policy: default DROP, then narrow allows ----
iptables -F
iptables -X 2>/dev/null || true

# Loopback is always fine.
iptables -A INPUT  -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow return traffic for connections we initiated.
iptables -A INPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow DNS to the container's configured resolver(s).
for dns in $DNS_SERVERS; do
    iptables -A OUTPUT -p udp --dport 53 -d "$dns" -j ACCEPT
    iptables -A OUTPUT -p tcp --dport 53 -d "$dns" -j ACCEPT
done

# Allow HTTPS only to the resolved Anthropic API IPs.
iptables -A OUTPUT -p tcp --dport 443 -m set --match-set allowed-api dst -j ACCEPT

# Default deny everything else, both directions.
iptables -P INPUT   DROP
iptables -P OUTPUT  DROP
iptables -P FORWARD DROP

echo "[firewall] egress locked to: ${ALLOWED_HOSTS[*]} (DNS: ${DNS_SERVERS:-none})" >&2

# Fail-closed verification: a non-allowlisted host MUST fail. If it unexpectedly
# succeeds, abort the run rather than proceed with an open network.
if curl --silent --show-error --max-time 5 https://example.com >/dev/null 2>&1; then
    echo "[firewall] FATAL: reached example.com — egress NOT locked, aborting." >&2
    exit 1
fi
echo "[firewall] verified: non-allowlisted egress is blocked." >&2
