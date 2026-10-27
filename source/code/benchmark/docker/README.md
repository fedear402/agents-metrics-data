# R3 Full-Agent Docker Runtime

Isolated runtime for the `agentic` (R3) exec mode of the benchmark. A fresh,
ephemeral container per run mounts **only** the per-run working directory
(seeded with `input.csv`), runs `claude -p` with `--permission-mode
bypassPermissions` inside it, restricts outbound network to the Anthropic API
endpoint, and leaves `result.json` plus the agent's script in the workdir for
the harness to copy back and score.

This is the production-isolation path described in the 2026-06-25 R3 build
plan, section 2. It is the hardened replacement for the out-of-repo tempdir
used by the prototype.

> Status: authored for review. The image has **not** been built. Read the
> "Auth" and "Stata dependency" sections before building.

## Files

| File | Role |
|---|---|
| `Dockerfile` | Image: Python 3.14 + host-matched scientific stack, R 4.6 + econometrics packages, Claude Code CLI, egress-firewall tooling, inert Stata stub. |
| `requirements-pinned.txt` | Python stack pinned to the host reference versions (pandas 3.0.0 etc.). Single source of truth. |
| `install_r_packages.R` | R econometrics packages from a dated P3M snapshot. |
| `init-firewall.sh` | iptables/ipset egress allowlist: DNS + `api.anthropic.com` only. |
| `entrypoint.sh` | Runs firewall as root, drops to non-root `bench`, execs `claude` under a wall-clock timeout. |
| `run_agentic_container.sh` | Host wrapper: one ephemeral container per run. Drop-in for `harness.run_claude_agentic`. |
| `.dockerignore` | Keeps the build context to the COPYed files; keeps the Stata license out by default. |
| `stata/` | Empty stub dir; where the user drops the Stata installer + license to opt in. |

## Build

```bash
cd code/benchmark/docker
docker build -t agents-metrics-agentic:latest .
# Apple Silicon + Stata later: build for x86-64 (Stata Linux is amd64-only):
#   docker build --platform=linux/amd64 -t agents-metrics-agentic:latest .
```

Version pins worth knowing (verified on the host 2026-06-30): Python 3.14.6,
numpy 2.4.2, pandas 3.0.0, scipy 1.18.0, statsmodels 0.14.6, Claude Code 2.1.196,
R 4.6.0. The image installs R from the CRAN Debian repo (`trixie-cran40`) to
track the host's R 4.6.x; exact patch parity is best-effort, and if CRAN has not
published trixie binaries yet you may need an older `-cran40` codename or a
source build (see the Dockerfile note). `linearmodels` is **not** on the host and
is **not** used by the reference outputs; it is included in the image only as an
agent convenience (see the comments in `requirements-pinned.txt`).

## Run one cell

```bash
# 1. Mint a subscription token once on the host (keeps subscription billing,
#    so the harness PacingController's accounting stays valid):
claude setup-token
export CLAUDE_CODE_OAUTH_TOKEN=...        # from the command above

# 2. Prepare an isolated per-run workdir containing ONLY input.csv:
mkdir -p /tmp/agentic_run && cp /path/to/input.csv /tmp/agentic_run/

# 3. Launch (prompt on stdin):
./run_agentic_container.sh --workdir /tmp/agentic_run --timeout 1800 < prompt.md
```

After it exits, `/tmp/agentic_run/` holds `result.json`, `claude_stream.jsonl`
(cost / tool-call telemetry), `claude_stderr.txt`, and the agent's script.

## Wiring into the harness

The wrapper deliberately mirrors the contract of `harness.run_claude_agentic()`
(prompt in, `result.json` + `run.*` out, stream-json telemetry). To switch the
`agentic` branch from the tempdir path to the container path:

- In `run_cell`'s `elif exec_mode == "agentic":` block (`harness.py` ~2109),
  replace the in-process `run_claude_agentic(...)` call with a
  `subprocess.run([...])` invocation of `run_agentic_container.sh`, passing the
  workdir produced by `prepare_agentic_dir(cell_id)` and the prompt via stdin.
- Keep `copy_back(workdir, cell_dir)` as-is — it already normalizes the agent's
  script to `run.{py,R,do}` and copies `result.json` back, so the existing
  `score()` and `score_integrity()` (`strict_posthoc.py`) run unchanged.
- Parse `claude_stream.jsonl` for `total_cost_usd` / `num_turns` / tool-call
  count exactly as `run_claude_agentic` already does (`load_json_lines`,
  `final_stream_result`, `recursive_tool_use_count`), and store the result under
  the record key `claude` so the cost/token post-hoc picks it up.
- **Run under the PacingController.** One paced run launches one container,
  sequentially. The container adds no concurrency; pacing headroom on the
  subscription is preserved exactly as in the other interface conditions.

## Auth: two approaches and their tradeoffs

`claude -p` inside the container needs credentials. The container never mounts
the repo, so credentials must be passed in explicitly. The wrapper supports two
modes via `--auth`.

### `oauth` (default, recommended) — subscription token

```bash
claude setup-token                 # mints a long-lived OAuth token on the host
export CLAUDE_CODE_OAUTH_TOKEN=... # wrapper passes it with `docker run -e`
./run_agentic_container.sh --auth oauth --workdir ...
```

- **Billing:** the **subscription**, as in every other interface condition. This keeps the
  PacingController's subscription accounting meaningful and supports the hard
  constraint that R3 must not starve the user's other interactive Claude usage.
- **Why a token, not a mounted credential file:** on this macOS host the
  subscription OAuth credential lives in the **login Keychain**, not in a file,
  so there is no `~/.claude/.credentials.json` to bind-mount (confirmed: only
  config/settings JSON files exist under `~/.claude`). `claude setup-token`
  produces a portable token that works headless in Linux. On a **Linux** host
  the credential *is* a file and could instead be mounted read-only
  (`-v "$HOME/.claude/.credentials.json":/home/bench/.claude/.credentials.json:ro`),
  but the token approach is uniform across hosts and is preferred.
- **Tradeoffs / risks:** the token is a bearer secret with the subscription's
  privileges. It is passed as an env var (visible in that container's
  `docker inspect` while it runs, and to any process that can read the host
  environment). Mitigations in place: it is never written to disk by the
  wrapper, never baked into the image, the container is `--rm` (ephemeral),
  network is locked to the Anthropic API so the token cannot be exfiltrated to
  a third party, and the repo is not mounted so the agent cannot read it from
  project files. Rotate by re-running `claude setup-token` and revoking the old
  token from the Anthropic console.

### `apikey` — API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./run_agentic_container.sh --auth apikey --workdir ...
```

- **Billing:** **API credits**, a separate meter from the subscription. This
  **breaks** the pacing controller's subscription accounting — the controller
  governs subscription usage, not API spend. Use only if you intend R3 to bill
  the API and you budget that spend separately.
- **Tradeoffs:** same bearer-secret handling and mitigations as above. Simpler
  to provision (no `setup-token` step), but it is the wrong choice if R3 is
  meant to pool with the subscription-billed interface conditions.

> Do **not** mount the whole host `~/.claude` directory read-write into the
> container. It holds settings, MCP auth caches, and history; a
> bypassPermissions agent could mutate them. If you mount config at all, mount
> only what is needed and mount it `:ro`.

## Network restriction

`init-firewall.sh` (run by `entrypoint.sh` as root before privileges drop) sets
iptables policy to default-DROP and allows only: loopback, established/related
return traffic, DNS to the container resolver, and HTTPS to the resolved IPs of
`api.anthropic.com`. It then **fail-closed verifies** that a non-allowlisted host
(`example.com`) is unreachable and aborts the run if it is not.

The image disables Claude Code's non-essential traffic
(`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, `DISABLE_AUTOUPDATER=1`) so the
allowlist needs nothing beyond the API endpoint — no telemetry, error-reporting,
autoupdate, or package-registry egress at run time (the image is pre-baked).

Requires the container to have `NET_ADMIN`/`NET_RAW` (the wrapper adds them).
Without them, `entrypoint.sh` refuses to run unless `ALLOW_OPEN_NETWORK=1`
(`--open-network` on the wrapper) is set — debug only.

## Isolation guarantees (and limits)

What this runtime enforces, beyond the tempdir prototype:

- **No repo on the filesystem.** The only mount is the per-run workdir. The
  agent cannot reach `data/tasks/references/` (the answer key) by path traversal —
  it is not in the container at all. This directly addresses the
  reward-hacking / hidden-label risk noted in the build plan.
- **Network egress locked** to the Anthropic API (above).
- **Reduced privileges:** `--cap-drop=ALL` (then only NET_ADMIN/NET_RAW for the
  firewall), `--security-opt no-new-privileges`, non-root `bench` user runs the
  agent, capped memory / CPU / PIDs, ephemeral `--rm`, `tmpfs /tmp`.
- **Defense in depth still applies:** keep `score_integrity()` in
  `strict_posthoc.py` as the backstop — it flags reference-path access patterns
  and reference numeric literals in the agent's copied-back script, and verifies
  the `input.csv` sha256 matches the canonical data.

Limits: the OAuth/API token is necessarily inside the container while it runs;
bind-mount file ownership on a **Linux** host will reflect the container's
`bench` uid (1000) — harmless for copy-back but adjust if your host user differs
(on macOS Docker Desktop, ownership is virtualized and files appear as the host
user). The firewall trusts DNS resolution at container start; the allowlist is
IP-based for the lifetime of the run.

## Stata dependency (what you must provide)

The user chose "all 3 platforms in Docker (license Stata in the image)". Stata
is proprietary, x86-64-only, and node-locked; **nothing in this directory
downloads, installs, or licenses it.** The Dockerfile contains a clearly-marked,
commented, inert Stata block. To enable the Stata platform you must supply, and
StataCorp licensing terms must permit, the following:

1. **Linux installer tarball.** `Stata<NN>Linux64.tar.gz` for your edition
   (Stata/BE, /SE, or /MP), e.g. `Stata18Linux64.tar.gz`. Downloadable only from
   StataCorp with a valid license (stata.com -> My Stata -> Download). Stata for
   Linux is **x86-64 only** — on Apple Silicon, build and run the image as
   `--platform=linux/amd64` (emulated), or build on an x86-64 host. Place it at
   `code/benchmark/docker/stata/Stata<NN>Linux64.tar.gz`.

2. **A valid Linux license.** Either (a) a pre-activated `stata.lic` file to copy
   into `/usr/local/stata<NN>/`, or (b) the **serial number, code, and
   authorization** from your Stata License & Activation Key, used to run
   `./stinit` during the build. Note: a **Linux** license is required — the
   macOS license on the host (`/Applications/StataNow/...`) does **not** cover a
   Linux container.

3. **StataCorp's written confirmation that the license permits this
   deployment.** Standard single-user Stata licenses are node-locked to one
   physical machine; running in a container, on a different/virtual host, or
   across multiple ephemeral containers generally requires a different license
   class and must be cleared with StataCorp licensing first. This is a legal
   dependency to surface, not a technical one to work around.

Once you have (1)-(3): add the artifacts under `stata/`, un-ignore them in
`.dockerignore`, uncomment the Stata block in the `Dockerfile` (set
`ARG STATA_VERSION` to your edition), and rebuild. The block extracts the
tarball to `/usr/local/stata<NN>`, runs `./install`, applies the license
(`stata.lic` or `./stinit`), and puts the console binary on `PATH`. The harness
calls `stata-se -b do run.do`; ensure the matching console binary
(`stata`/`stata-se`/`stata-mp`) exists and set the harness `STATA_BIN`/PATH for
the in-container path accordingly. **Never push or share the resulting image** —
it embeds a node-locked license.
