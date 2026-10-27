"""Worker preflight: verify the local stack matches the canonical benchmark.

Checks, in one pass:
- R: the canonical package set parsed at runtime from
  ``docker/install_r_packages.R`` (12 packages) plus ``survival`` (ships with
  base R), via a single ``Rscript --vanilla`` call that prints versions.
- Python: numpy/pandas/scipy/statsmodels/linearmodels importable in THIS
  interpreter (``sys.executable``). IMPORTANT: run preflight with the same
  python the harness uses to execute generated code, e.g. the docker/venv
  interpreter, not the repo tooling venv.
- Stata: batch binary existence via ``harness.detect_stata_bin()``.
- claude CLI: presence + version. The pinned-model resolution probe
  (``claude -p ... --model claude-sonnet-4-6``) runs ONLY under
  ``--live-model-check`` so the default preflight spends no tokens.
- OS + hostname.

Output is the capabilities payload for ``POST /v1/preflight``:
{os, hostname, platforms: {python: {ok, versions}, r: {ok, versions},
 stata: {ok, path}}, claude: {version}}.

Strict mode (``--strict``) exits 1 when a canonical package is missing on a
platform this worker claims. A platform is claimed when its runtime is
present (python always; R iff Rscript is on PATH; Stata iff the binary
exists) — a wholly absent runtime just means the worker will not claim that
platform's cells. A missing claude CLI or a failed live model probe is always
a strict failure.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from queue_client import QueueClient, QueueError  # noqa: E402

R_INSTALL_SCRIPT = HERE / "docker" / "install_r_packages.R"
# survival ships with base R and is deliberately absent from the docker
# installer; workers must still have it loadable.
EXTRA_R_PACKAGES = ("survival",)
# lifelines added 2026-07-06: it is a ratified "available standard package"
# (decision record 2026-06-30) and the legacy environment had it; the pinned
# worker lacking it produced 3 spurious cox_hazard python failures (audit
# 2026-07-06) — models burned self-repair attempts on the missing import.
PYTHON_PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "linearmodels", "lifelines")
PINNED_MODEL = "claude-sonnet-4-6"
# Subprocess guards for the local checks (client-side policy values).
R_CHECK_TIMEOUT_SECONDS = 300
CLAUDE_VERSION_TIMEOUT_SECONDS = 60
MODEL_CHECK_TIMEOUT_SECONDS = 240


def parse_r_packages(path: Path = R_INSTALL_SCRIPT) -> list[str]:
    """Extract the ``pkgs <- c(...)`` vector from the R installer script."""
    text = path.read_text()
    match = re.search(r"pkgs\s*<-\s*c\((.*?)\)", text, re.DOTALL)
    if match is None:
        raise ValueError(f"no `pkgs <- c(...)` vector found in {path}")
    return re.findall(r'"([^"]+)"', match.group(1))


def canonical_r_packages() -> list[str]:
    """Installer packages plus the base-R extras the tasks rely on."""
    pkgs = parse_r_packages()
    return pkgs + [extra for extra in EXTRA_R_PACKAGES if extra not in pkgs]


def check_r(pkgs: list[str] | None = None) -> dict[str, Any]:
    """One Rscript --vanilla call printing `<pkg> <version|MISSING>` lines."""
    if pkgs is None:
        pkgs = canonical_r_packages()
    rscript = shutil.which("Rscript")
    if rscript is None:
        return {
            "ok": False,
            "versions": {},
            "missing": list(pkgs),
            "error": "Rscript not found",
        }
    quoted = ", ".join(f'"{pkg}"' for pkg in pkgs)
    expr = (
        f"pkgs <- c({quoted}); "
        'cat("R", as.character(getRversion()), "\\n"); '
        "for (p in pkgs) { "
        "v <- tryCatch(as.character(utils::packageVersion(p)), "
        "error = function(e) NA_character_); "
        'cat(p, ifelse(is.na(v), "MISSING", v), "\\n") }'
    )
    try:
        proc = subprocess.run(
            [rscript, "--vanilla", "-e", expr],
            capture_output=True,
            text=True,
            timeout=R_CHECK_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        return {"ok": False, "versions": {}, "missing": list(pkgs), "error": str(err)}
    versions: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            versions[parts[0]] = parts[1]
    missing = [pkg for pkg in pkgs if versions.get(pkg) in (None, "MISSING")]
    result: dict[str, Any] = {
        "ok": proc.returncode == 0 and not missing,
        "versions": versions,
        "missing": missing,
    }
    if proc.returncode != 0:
        result["error"] = (proc.stderr or "").strip()[:300]
    return result


def check_python() -> dict[str, Any]:
    """Import the canonical stack in THIS interpreter (sys.executable)."""
    versions: dict[str, str] = {"python": platform.python_version()}
    missing: list[str] = []
    for pkg in PYTHON_PACKAGES:
        try:
            module = importlib.import_module(pkg)
        except Exception:  # ImportError or any import-time failure
            versions[pkg] = "MISSING"
            missing.append(pkg)
            continue
        versions[pkg] = str(getattr(module, "__version__", "unknown"))
    return {
        "ok": not missing,
        "versions": versions,
        "missing": missing,
        "executable": sys.executable,
    }


def check_stata() -> dict[str, Any]:
    """Existence of the batch binary resolved by harness.detect_stata_bin().

    ``is_statanow`` reports whether the resolved binary is a StataNow install.
    The recorded worker environment is StataNow/SE 19.5; a leaked ``STATA_BIN``
    pointing at perpetual Stata 14 produced the 2026-07-14 stopgap records
    (re-collected by the retired ``recollect_stata19.sh``, whose version guard
    moved here on 2026-08-18).
    """
    try:
        from harness import detect_stata_bin
    except Exception as err:  # keep preflight usable even if harness breaks
        return {"ok": False, "path": None, "error": f"harness import failed: {err}"}
    path = detect_stata_bin()
    return {
        "ok": path.exists(),
        "path": str(path),
        "is_statanow": "StataNow" in str(path),
    }


def check_claude(live_model_check: bool = False) -> dict[str, Any]:
    """claude CLI presence + version; token-spending probe is opt-in."""
    exe = shutil.which("claude")
    if exe is None:
        return {"ok": False, "version": None, "error": "claude CLI not found"}
    try:
        proc = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=CLAUDE_VERSION_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        return {"ok": False, "version": None, "error": str(err)}
    version = (proc.stdout or proc.stderr).strip()
    result: dict[str, Any] = {"ok": proc.returncode == 0, "version": version or None}
    if not live_model_check:
        result["model_check"] = {"model": PINNED_MODEL, "skipped": True}
        return result
    try:
        probe = subprocess.run(
            [exe, "-p", "Reply with the single word OK.", "--model", PINNED_MODEL],
            capture_output=True,
            text=True,
            timeout=MODEL_CHECK_TIMEOUT_SECONDS,
            check=False,
        )
        probe_ok = probe.returncode == 0
        output = (probe.stdout or probe.stderr).strip()[:200]
    except (OSError, subprocess.TimeoutExpired) as err:
        probe_ok, output = False, str(err)
    result["model_check"] = {"model": PINNED_MODEL, "ok": probe_ok, "output": output}
    result["ok"] = result["ok"] and probe_ok
    return result


def run_checks(live_model_check: bool = False) -> dict[str, Any]:
    return {
        "python": check_python(),
        "r": check_r(),
        "stata": check_stata(),
        "claude": check_claude(live_model_check),
    }


def build_payload(checks: dict[str, Any]) -> dict[str, Any]:
    """Capabilities payload matching the contract's /v1/preflight body.

    Minimized: sends capability + versions (needed for the parity check) but not
    the absolute Stata path (can embed a username) — presence is enough. Set
    AGENTS_METRICS_NO_HOSTNAME=1 to also omit the hostname.
    """
    payload: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "platforms": {
            "python": {
                "ok": checks["python"]["ok"],
                "versions": checks["python"]["versions"],
            },
            "r": {"ok": checks["r"]["ok"], "versions": checks["r"]["versions"]},
            "stata": {"ok": checks["stata"]["ok"]},
        },
        "claude": {"version": checks["claude"].get("version")},
    }
    if not os.environ.get("AGENTS_METRICS_NO_HOSTNAME"):
        payload["hostname"] = socket.gethostname()
    return payload


def strict_failures(checks: dict[str, Any]) -> list[str]:
    """Failures that make a worker unfit for the platforms it claims."""
    failures: list[str] = []
    python_check = checks["python"]
    if python_check.get("missing"):
        failures.append(
            "python: missing packages: " + ", ".join(python_check["missing"])
        )
    r_check = checks["r"]
    if r_check.get("error") == "Rscript not found":
        pass  # platform not claimed by this worker
    elif r_check.get("missing"):
        failures.append("r: missing packages: " + ", ".join(r_check["missing"]))
    elif not r_check["ok"]:
        failures.append("r: Rscript check failed: " + str(r_check.get("error", "")))
    # stata: a missing binary only means this worker does not claim stata.
    claude_check = checks["claude"]
    if claude_check.get("version") is None:
        failures.append("claude: CLI not available")
    if claude_check.get("model_check", {}).get("ok") is False:
        failures.append(f"claude: pinned model {PINNED_MODEL} probe failed")
    return failures


def print_human(
    checks: dict[str, Any], payload: dict[str, Any], failures: list[str]
) -> None:
    print(f"os: {payload['os']}   hostname: {payload.get('hostname', '(omitted)')}")
    python_check = checks["python"]
    print(
        f"python [{'ok' if python_check['ok'] else 'FAIL'}] "
        f"{python_check['executable']} "
        + " ".join(f"{k}={v}" for k, v in python_check["versions"].items())
    )
    r_check = checks["r"]
    r_line = f"r      [{'ok' if r_check['ok'] else 'FAIL'}] "
    if r_check.get("error"):
        r_line += r_check["error"]
    else:
        r_line += " ".join(f"{k}={v}" for k, v in r_check["versions"].items())
    print(r_line)
    stata_check = checks["stata"]
    print(
        f"stata  [{'ok' if stata_check['ok'] else 'FAIL'}] "
        f"{stata_check.get('path') or stata_check.get('error', '')}"
    )
    claude_check = checks["claude"]
    model_check = claude_check.get("model_check", {})
    if model_check.get("skipped"):
        model_note = f"model check skipped (pinned: {PINNED_MODEL})"
    else:
        model_state = "ok" if model_check.get("ok") else "FAIL"
        model_note = f"model {model_check.get('model')} probe: {model_state}"
    print(
        f"claude [{'ok' if claude_check['ok'] else 'FAIL'}] "
        f"{claude_check.get('version') or claude_check.get('error', '')} — {model_note}"
    )
    if failures:
        print("failures:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("failures: none")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 on any strict failure"
    )
    parser.add_argument(
        "--post", action="store_true", help="report capabilities to the API"
    )
    parser.add_argument("--json", action="store_true", help="print JSON output")
    parser.add_argument(
        "--live-model-check",
        action="store_true",
        help=f"actually invoke `claude -p --model {PINNED_MODEL}` (spends tokens)",
    )
    args = parser.parse_args(argv)

    checks = run_checks(live_model_check=args.live_model_check)
    payload = build_payload(checks)
    failures = strict_failures(checks)

    post_response: dict[str, Any] | None = None
    post_error: str | None = None
    if args.post:
        try:
            post_response = QueueClient().preflight(payload)
        except QueueError as err:
            post_error = str(err)

    if args.json:
        document = {
            "payload": payload,
            "checks": checks,
            "failures": failures,
        }
        if args.post:
            document["post_response"] = post_response
            document["post_error"] = post_error
        print(json.dumps(document, indent=2))
    else:
        print_human(checks, payload, failures)
        if args.post:
            if post_error:
                print(f"post: FAILED — {post_error}")
            else:
                print(f"post: ok — {json.dumps(post_response)}")

    if post_error:
        return 1
    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
