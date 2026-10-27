# syntax=docker/dockerfile:1
# ============================================================================
# R3 full-agent runtime image
# Project: Chatbots, Agents, and the Choice of Econometric Software
# ----------------------------------------------------------------------------
# What this image is: an isolated runtime where `claude -p` runs with
# --permission-mode bypassPermissions inside an ephemeral container that mounts
# ONLY a per-run working directory (input.csv). It is the production-isolation
# replacement for the out-of-repo tempdir prototype described in the
# 2026-06-25 R3 build plan.
#
# Contents:
#   - Python 3.14 + the host-matched scientific stack (pandas 3.0.0 etc.)
#   - R 4.6 + common econometrics packages
#   - Claude Code CLI (pinned)
#   - egress firewall tooling (iptables/ipset) so the wrapper can lock network
#     down to the Anthropic API endpoint only
#   - a CLEARLY-MARKED, INERT Stata stub (no Stata is downloaded or installed)
#
# NOTE: this file is authored for review and is NOT built by the agent that
# wrote it. Build it yourself (see README.md). Do not `docker build` from CI
# without reading the Stata and auth sections first.
#
# Apple-Silicon note: build for linux/amd64 if you intend to add Stata (Stata
# for Linux is x86-64 only). The Python/R/Claude layers build natively on arm64.
# ============================================================================

# Python minor version matches the host (3.14.6). Trixie base gives a modern
# toolchain for compiling R source packages.
FROM python:3.14-slim-trixie

ARG DEBIAN_FRONTEND=noninteractive
# Pin the Claude Code CLI to the host version for parity (host: 2.1.196).
ARG CLAUDE_CODE_VERSION=2.1.196
# Node major used only to host the Claude Code CLI.
ARG NODE_MAJOR=20
# Dated R package snapshot (see install_r_packages.R).
ARG P3M_SNAPSHOT=2026-06-01

# ----------------------------------------------------------------------------
# 1a. Add the CRAN Debian apt repo so r-base tracks the host R (4.6.x).
#     Debian's own r-base often lags; CRAN ships current R. If CRAN has not yet
#     published trixie binaries, swap `trixie-cran40` for the latest available
#     codename (e.g. bookworm-cran40) or build R from source.
# ----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg dirmngr; \
    install -d /usr/share/keyrings; \
    curl -fsSL https://cloud.r-project.org/bin/linux/debian/marutter_pubkey.asc \
        | gpg --dearmor -o /usr/share/keyrings/cran.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/cran.gpg] https://cloud.r-project.org/bin/linux/debian trixie-cran40/" \
        > /etc/apt/sources.list.d/cran.list

# ----------------------------------------------------------------------------
# 1b. System packages: build toolchain (for R source pkgs), R itself, network
#     tooling for the egress firewall, and gosu for dropping privileges.
# ----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg dirmngr git \
        build-essential gfortran cmake pkg-config \
        # R + its common build dependencies (r-base from the CRAN repo above)
        r-base r-base-dev \
        libcurl4-openssl-dev libssl-dev libxml2-dev \
        libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
        libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
        libblas-dev liblapack-dev libgsl-dev \
        # egress-firewall + diagnostics (used by init-firewall.sh)
        iptables ipset iproute2 dnsutils \
        # privilege drop + process tools
        gosu procps; \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# 2. Node.js (NodeSource) + Claude Code CLI, pinned.
# ----------------------------------------------------------------------------
RUN set -eux; \
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -; \
    apt-get install -y --no-install-recommends nodejs; \
    npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"; \
    rm -rf /var/lib/apt/lists/* /root/.npm; \
    claude --version

# ----------------------------------------------------------------------------
# 3. Python scientific stack, pinned to the host reference versions.
#    Installed with uv into the image's system interpreter.
# ----------------------------------------------------------------------------
RUN pip install --no-cache-dir uv
COPY requirements-pinned.txt /tmp/requirements-pinned.txt
RUN uv pip install --system --no-cache -r /tmp/requirements-pinned.txt; \
    python -c "import numpy, pandas, scipy, statsmodels; \
print('python stack:', numpy.__version__, pandas.__version__, scipy.__version__, statsmodels.__version__)"

# ----------------------------------------------------------------------------
# 4. R econometrics packages (from a dated P3M snapshot).
# ----------------------------------------------------------------------------
COPY install_r_packages.R /tmp/install_r_packages.R
RUN P3M_SNAPSHOT="${P3M_SNAPSHOT}" Rscript /tmp/install_r_packages.R

# ============================================================================
# 5. STATA STUB — INERT. Nothing here downloads, installs, or licenses Stata.
# ============================================================================
# The user chose "all 3 platforms in Docker (license Stata in the image)".
# Stata for Linux is proprietary, x86-64 only, and node-locked; it CANNOT be
# fetched here. To enable the Stata platform, the user must supply the installer
# tarball and a valid Linux license, then UNCOMMENT and complete the block
# below. See README.md -> "Stata dependency (what you must provide)".
#
# Expected layout under the build context before uncommenting:
#   code/benchmark/docker/stata/Stata<NN>Linux64.tar.gz   (from StataCorp)
#   code/benchmark/docker/stata/stata.lic                 (activated license)
#
# ARG STATA_VERSION=18
# COPY stata/Stata${STATA_VERSION}Linux64.tar.gz /tmp/stata.tar.gz
# RUN set -eux; \
#     mkdir -p /usr/local/stata${STATA_VERSION}; \
#     tar -xzf /tmp/stata.tar.gz -C /usr/local/stata${STATA_VERSION}; \
#     cd /usr/local/stata${STATA_VERSION}; \
#     yes | ./install || true; \
#     rm -f /tmp/stata.tar.gz
# # Option A: drop in a pre-activated license file.
# COPY stata/stata.lic /usr/local/stata${STATA_VERSION}/stata.lic
# # Option B (instead of A): activate at build with ./stinit using the serial /
# #   code / authorization from the user's Stata License & Activation Key.
# #   This is interactive; script it with `expect` or a heredoc to ./stinit.
# ENV PATH="/usr/local/stata${STATA_VERSION}:${PATH}"
# # The harness calls `stata-se -b do run.do`; ensure the matching console
# # binary (stata / stata-se / stata-mp) exists on PATH and set STATA_BIN
# # accordingly when wiring the harness to run inside the container.
# RUN command -v stata-se || command -v stata-mp || command -v stata
# ============================================================================

# ----------------------------------------------------------------------------
# 6. Non-root run user + Claude config dir.
# ----------------------------------------------------------------------------
RUN useradd --create-home --uid 1000 --shell /bin/bash bench; \
    mkdir -p /home/bench/.claude /work; \
    chown -R bench:bench /home/bench /work
ENV HOME=/home/bench

# Quiet, network-minimal Claude Code: no autoupdate / telemetry / error-reporting
# so the egress firewall only has to allow the Anthropic API (+ DNS).
ENV CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    DISABLE_AUTOUPDATER=1

# ----------------------------------------------------------------------------
# 7. Entrypoint + firewall.
# ----------------------------------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/init-firewall.sh

WORKDIR /work
# entrypoint.sh runs the firewall as root (if NET_ADMIN is granted) and then
# drops to the `bench` user to exec whatever command the wrapper passes
# (normally: claude -p ...). The agentic prompt is fed on stdin.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["claude", "--version"]
