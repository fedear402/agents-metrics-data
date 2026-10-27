"""Watch the Claude Max and ChatGPT Pro usage windows and pace a collection.

One watcher state serves both providers. For each provider, the watcher keeps the latest
usage of each limit window and returns a decision before each new provider request:

- ``run``: start the request.
- ``slow``: start the request, but keep a chosen spacing between request starts.
- ``pause``: wait until the window reset time plus a chosen margin, then decide again.
- ``retry``: a Claude pause without a usable reset time (for example a rejection that names no reset).
  One request may start every ``unknown_reset_retry_seconds`` (chosen: 900 s). Its stream gives the next
  reading. Extra usage is off on the Claude account, so a request past the window is rejected, not billed.

Sources. Neither source spends model tokens.

- Claude: the ``rate_limit_event`` lines that Claude Code writes into each provider stream
  of the collection (five-hour and seven-day utilization, reset times, extra-usage flags).
  Documentation: https://code.claude.com/docs/en/agent-sdk/python#ratelimitinfo
- Codex: the ``account/rateLimits/read`` request of ``codex app-server``. It reads the
  account windows and sends no model request. ``codex exec --json`` streams carry no limit
  data. Local Codex session files (``~/.codex/sessions``) are a fallback for ``status``.
  The answer has one view per metered limit bucket (``rateLimitsByLimitId``). The watcher
  reads every bucket. Windows of the ``codex`` bucket keep their plain names (``seven_day``);
  windows of another bucket are named ``<limit id>:<window>``. Decisions use the ``codex``
  bucket and any bucket whose model slug is a collection model (``model_slugs``).
  Validation (2026-09-26, quality_reports/audits/2026-09-26_rerun-design/work/watcher-validation):
  the app-server reading, the per-call ``token_count`` readings of non-ephemeral sessions,
  and the isolated-home reading agree. The percent is an integer. One Luna collection call
  costs far less than one point of the week, so a batch of a few hundred calls does not move it.

The thresholds are chosen limits in ``usage_watch.json``. They are not provider values.
The watcher never uses a Codex rate-limit reset credit and never turns on extra usage.

Commands (run from the project root):

  status [--runs DIR ...] [--no-poll] [--json]   show and record both providers
  replay --runs DIR [--limit N]                   decisions from saved streams, no calls
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name('usage_watch.json')
VERSION = 'usage_watch_2026_10_02_v1'
# Hard pause level that no configuration or caller override can raise. At 100 percent of a Codex
# window the account answers from its credit balance (seen on 2026-10-02), and Federico forbids any
# credit use, so Codex work stops at 99 percent. Claude extra usage is off, so a full Claude window rejects.
HARD_PAUSE_AT = {'codex': 0.99}
DOC = 'https://code.claude.com/docs/en/agent-sdk/python#ratelimitinfo'
# The only JSON-RPC requests sent to codex app-server. None of them starts a model turn.
CODEX_REQUESTS = ('initialize', 'initialized', 'account/rateLimits/read')
# Reset times after this horizon are treated as unknown (seconds, chosen check).
MAX_RESET_HORIZON_SECONDS = 8 * 24 * 60 * 60
# Two readings whose reset times differ by at most this many seconds describe one window
# (chosen tolerance). Codex readings of one window differ by 1 to 2 seconds (2026-09-26).
# Distinct windows of one bucket are at least five hours apart.
RESET_JITTER_SECONDS = 600
# Codex limit bucket that meters all models when no model-specific bucket applies.
CODEX_MAIN_LIMIT = 'codex'
DECISIONS = ('run', 'slow', 'pause', 'retry')
# A Claude result error with this text is a usage-limit rejection (extra usage is off, so the call is refused).
LIMIT_TEXT = re.compile(r'usage limit|hit your limit|limit reached|out of (extra )?usage', re.I)
RESET_IN_TEXT = re.compile(r'\|(\d{10})\b')


class WatchStopped(RuntimeError):
    """The collection stop flag was set while a request waited."""


class WatchUnknownReset(RuntimeError):
    """Kept for imports only. Since usage_watch_2026_09_29_v1 the gate never raises it: a Claude pause
    without a reset time retries one call every ``unknown_reset_retry_seconds``."""


class PollError(RuntimeError):
    """The Codex status request failed."""


def load_config(path: Path | None = None) -> dict:
    """Return the chosen pacing limits."""
    return json.loads(Path(path or CONFIG_PATH).read_text())


def provider_for(model: str | None, config: dict) -> str | None:
    """Map a collector model id (``sonnet``, ``luna``) or a provider name to a provider."""
    if model in config['providers']:
        return model
    for provider, settings in config['providers'].items():
        if model in settings.get('models', []):
            return provider
    return None


def iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _reset(value: Any, observed_at: float) -> float | None:
    """Accept a reset time in epoch seconds inside a plausible horizon."""
    value = _number(value)
    if value is None or not -MAX_RESET_HORIZON_SECONDS < value - observed_at <= MAX_RESET_HORIZON_SECONDS:
        return None
    return value


def window_name(minutes: Any) -> str:
    minutes = _number(minutes)
    if minutes == 300:
        return 'five_hour'
    if minutes == 10080:
        return 'seven_day'
    return f'{int(minutes)}_minutes' if minutes is not None else 'unknown_window'


# Parsing

def claude_observation(events: Iterable[dict], observed_at: float, source: str) -> dict | None:
    """Build one observation from the last ``rate_limit_event`` of a Claude stream."""
    info = None
    for event in events:
        if isinstance(event, dict) and event.get('type') == 'rate_limit_event':
            candidate = event.get('rate_limit_info', event.get('rateLimitInfo'))
            if isinstance(candidate, dict):
                info = candidate
    rejected_text, text_reset = None, None
    for event in events:
        if isinstance(event, dict) and event.get('type') == 'result' and event.get('is_error'):
            message = str(event.get('result') or event.get('error') or '')
            if LIMIT_TEXT.search(message):
                rejected_text = message[:300]
                match = RESET_IN_TEXT.search(message)
                text_reset = _reset(int(match.group(1)), observed_at) if match else None
    if info is None and rejected_text is None:
        return None
    info = dict(info or {})
    if rejected_text is not None:
        info['status'] = 'rejected'
        if text_reset is not None and all(info.get(k) is None for k in ('resetsAt', 'resetAt', 'resets_at')):
            info['resetsAt'] = text_reset
    windows = {}
    for name, window in (info.get('unifiedWindows') or {}).items():
        if isinstance(window, dict):
            windows[name] = dict(used=_number(window.get('utilization')),
                                 resets_at=_reset(window.get('resetsAt'), observed_at))
    top_type = info.get('rateLimitType')
    top_reset = _reset(next((info[k] for k in ('resetsAt', 'resetAt', 'resets_at') if info.get(k) is not None), None),
                       observed_at)
    if top_type and top_type not in windows and _number(info.get('utilization')) is not None:
        windows[top_type] = dict(used=_number(info.get('utilization')), resets_at=top_reset)
    return dict(provider='claude', observed_at=observed_at, source=source, status=info.get('status'),
                rate_limit_type=top_type, rejected_resets_at=top_reset if info.get('status') == 'rejected' else None,
                is_using_overage=info.get('isUsingOverage'), overage_status=info.get('overageStatus'),
                rejection_text=rejected_text, windows=windows)


def _snapshot_value(snapshot: dict, camel: str, snake: str) -> Any:
    return snapshot.get(camel, snapshot.get(snake))


def limit_of(snapshot: dict) -> str:
    """Limit bucket id of one snapshot. A snapshot without an id is the main ``codex`` bucket."""
    return _snapshot_value(snapshot, 'limitId', 'limit_id') or CODEX_MAIN_LIMIT


def window_key(limit: str, name: str) -> str:
    """Window name in an observation. Windows of the main bucket keep their plain name."""
    return name if limit == CODEX_MAIN_LIMIT else f'{limit}:{name}'


def split_window_key(key: str) -> tuple[str, str]:
    limit, _, name = key.rpartition(':')
    return limit or CODEX_MAIN_LIMIT, name


def _codex_snapshots(result: Any) -> list[dict]:
    """All limit-bucket snapshots in an app-server result, a session payload, or a list of payloads."""
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if not isinstance(result, dict):
        return []
    by_id = result.get('rateLimitsByLimitId')
    single = result.get('rateLimits') if isinstance(result.get('rateLimits'), dict) else None
    if isinstance(by_id, dict) and by_id:
        snapshots = [dict(snapshot, limitId=_snapshot_value(snapshot, 'limitId', 'limit_id') or key)
                     for key, snapshot in by_id.items() if isinstance(snapshot, dict)]
        if single and limit_of(single) not in {limit_of(snapshot) for snapshot in snapshots}:
            snapshots.append(single)
        return snapshots
    return [single if single is not None else result]


def codex_observation(result: Any, observed_at: float, source: str) -> dict | None:
    """Build one observation from an app-server result or session ``rate_limits`` payloads.

    Every limit bucket is read. Each bucket keeps its own windows, so a second bucket's
    weekly window never replaces the weekly window of the main ``codex`` bucket.
    """
    windows, buckets = {}, {}
    main = None
    for snapshot in _codex_snapshots(result):
        limit = limit_of(snapshot)
        for slot in ('primary', 'secondary'):
            window = snapshot.get(slot)
            if not isinstance(window, dict):
                continue
            used = _number(_snapshot_value(window, 'usedPercent', 'used_percent'))
            name = window_name(_snapshot_value(window, 'windowDurationMins', 'window_minutes'))
            windows[window_key(limit, name)] = dict(
                used=None if used is None else round(used / 100, 6),
                resets_at=_reset(_snapshot_value(window, 'resetsAt', 'resets_at'), observed_at))
        buckets[limit] = dict(
            model_slug=_snapshot_value(snapshot, 'normalModelSlug', 'normal_model_slug'),
            rate_limit_reached_type=_snapshot_value(snapshot, 'rateLimitReachedType', 'rate_limit_reached_type'),
            spend_control_reached=_snapshot_value(snapshot, 'spendControlReached', 'spend_control_reached'))
        if main is None or limit == CODEX_MAIN_LIMIT:
            main = snapshot  # account-level fields come from the main bucket when present
    if not windows:
        return None
    wrapper = result if isinstance(result, dict) else {}
    credits = main.get('credits') if isinstance(main.get('credits'), dict) else {}
    reset_credits = wrapper.get('rateLimitResetCredits') if isinstance(wrapper.get('rateLimitResetCredits'), dict) else {}
    return dict(provider='codex', observed_at=observed_at, source=source,
                plan_type=_snapshot_value(main, 'planType', 'plan_type'),
                ordinary_usage_allowed=wrapper.get('ordinaryUsageAllowed'),
                rate_limit_reached_type=_snapshot_value(main, 'rateLimitReachedType', 'rate_limit_reached_type'),
                spend_control_reached=_snapshot_value(main, 'spendControlReached', 'spend_control_reached'),
                has_credits=_snapshot_value(credits, 'hasCredits', 'has_credits'),
                reset_credits_available=reset_credits.get('availableCount'),
                buckets=buckets, windows=windows)


def _find_rate_limits(value: Any) -> dict | None:
    if isinstance(value, dict):
        found = value.get('rate_limits')
        if isinstance(found, dict) and ('primary' in found or 'secondary' in found):
            return found
        for item in value.values():
            found = _find_rate_limits(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_rate_limits(item)
            if found:
                return found
    return None


def observation_from_events(provider: str, events: list, observed_at: float, source: str) -> dict | None:
    """Read limit data from a saved provider stream (the last payload of each limit bucket)."""
    if provider == 'claude':
        return claude_observation(events, observed_at, source)
    latest = {}
    for event in events:
        snapshot = _find_rate_limits(event)
        if snapshot:
            latest[limit_of(snapshot)] = snapshot
    return codex_observation(list(latest.values()), observed_at, source) if latest else None


def read_events(path: Path) -> list:
    events = []
    for line in Path(path).read_text(errors='replace').splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def merge(old: dict | None, new: dict) -> dict:
    """Merge two observations of one provider.

    Usage within one window only rises until its reset, so a late-arriving older reading of
    the same window cannot lower it. Reset times within ``RESET_JITTER_SECONDS`` describe
    one window; the reset time of the newer reading is kept. A reset time later by more than
    the tolerance marks a newer window.
    """
    if not old:
        return new
    new_is_newer = new['observed_at'] >= old['observed_at']
    newest = new if new_is_newer else old
    merged = dict(newest)
    windows = {}
    for name in set(old.get('windows', {})) | set(new.get('windows', {})):
        a, b = old.get('windows', {}).get(name), new.get('windows', {}).get(name)
        if not a or not b:
            windows[name] = a or b
            continue
        ra, rb = a.get('resets_at'), b.get('resets_at')
        if ra is not None and rb is not None and abs(ra - rb) > RESET_JITTER_SECONDS:
            windows[name] = a if ra > rb else b
            continue
        used = [x['used'] for x in (a, b) if x.get('used') is not None]
        first, second = (rb, ra) if new_is_newer else (ra, rb)
        windows[name] = dict(used=max(used) if used else None, resets_at=first if first is not None else second)
    merged['windows'] = windows
    if 'buckets' in old or 'buckets' in new:
        # Keep a bucket's model slug when the newer reading (a session file) does not carry one.
        older = old if new_is_newer else new
        buckets = {limit: dict(info) for limit, info in (older.get('buckets') or {}).items()}
        for limit, info in (newest.get('buckets') or {}).items():
            slug = info.get('model_slug') or buckets.get(limit, {}).get('model_slug')
            buckets[limit] = dict(info, model_slug=slug)
        merged['buckets'] = buckets
    return merged


# Codex status request

def poll_codex(timeout: float, command: list | None = None) -> dict:
    """Read the Codex account windows through ``codex app-server``. No model request."""
    command = list(command or ['codex', 'app-server'])
    if not Path(command[0]).is_absolute():
        found = shutil.which(command[0])
        if not found:
            raise PollError(f'{command[0]} not found on PATH')
        command[0] = found
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True)
    except OSError as error:
        raise PollError(repr(error)) from error
    deadline = time.monotonic() + timeout
    lines: queue.Queue = queue.Queue()

    def reader() -> None:
        # A thread avoids select() on a buffered stream, which can hide a buffered line.
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    try:
        def send(message: dict) -> None:
            process.stdin.write(json.dumps(message) + '\n')
            process.stdin.flush()

        def answer(request_id: int) -> dict:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PollError('codex app-server did not answer in time')
                try:
                    line = lines.get(timeout=remaining)
                except queue.Empty:
                    continue
                if line is None:
                    raise PollError(f'codex app-server exited with code {process.poll()}')
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict) and message.get('id') == request_id:
                    if 'error' in message:
                        raise PollError(json.dumps(message['error'])[:300])
                    return message.get('result') or {}

        send(dict(jsonrpc='2.0', id=1, method=CODEX_REQUESTS[0],
                  params=dict(clientInfo=dict(name='agents_metrics_usage_watch', version=VERSION))))
        answer(1)
        send(dict(jsonrpc='2.0', method=CODEX_REQUESTS[1]))
        send(dict(jsonrpc='2.0', id=2, method=CODEX_REQUESTS[2]))
        return answer(2)
    except (BrokenPipeError, OSError) as error:
        raise PollError(repr(error)) from error
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass
        process.terminate()
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def latest_codex_session_snapshot(root: Path | None = None, newest_files: int = 20) -> dict | None:
    """Return the newest ``rate_limits`` payload of each limit bucket from local Codex session files.

    ``payload`` is a list with one payload per bucket. ``observed_at`` and ``source`` belong to
    the newest payload of the main ``codex`` bucket, or to the newest payload when that bucket
    is absent.
    """
    root = Path(root or Path.home() / '.codex/sessions')
    if not root.is_dir():
        return None
    files = []
    for folder, _, names in os.walk(root):
        for name in names:
            if name.startswith('rollout-') and name.endswith('.jsonl'):
                path = Path(folder) / name
                files.append((path.stat().st_mtime, path))
    best = {}
    for mtime, path in sorted(files, reverse=True)[:newest_files]:
        for line in path.read_text(errors='replace').splitlines():
            if '"rate_limits"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            payload = _find_rate_limits(event)
            when = event.get('timestamp') if isinstance(event, dict) else None
            try:
                at = datetime.fromisoformat(str(when).replace('Z', '+00:00')).timestamp()
            except ValueError:
                at = mtime
            if payload:
                limit = limit_of(payload)
                if limit not in best or at > best[limit][0]:
                    best[limit] = (at, payload, str(path))
    if not best:
        return None
    lead = best.get(CODEX_MAIN_LIMIT) or max(best.values(), key=lambda item: item[0])
    return dict(observed_at=lead[0], payload=[item[1] for item in best.values()], source=lead[2])


# Watcher

class Watcher:
    """Shared usage state for both providers.

    The state folder is shared by every collector process on the machine. A file lock
    protects ``state.json``. Each process also uses a thread lock.
    """

    def __init__(self, config: dict | None = None, state_dir: Path | None = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], Any] | None = None,
                 stop: threading.Event | None = None,
                 codex_poller: Callable[[float], dict | None] | None = None) -> None:
        self.config = config or load_config()
        self.config_sha256 = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        folder = Path(state_dir or self.config['state_dir'])
        self.state_dir = folder if folder.is_absolute() else ROOT / folder
        self.clock = clock
        self.stop = stop or threading.Event()
        self.sleep = sleep or (lambda seconds: self.stop.wait(seconds))
        command = self.config['providers']['codex'].get('command')
        self.codex_poller = codex_poller or (lambda timeout: poll_codex(timeout, command))
        self.lock = threading.RLock()
        self.poll_lock = threading.Lock()

    # State file

    @contextmanager
    def _state(self, write: bool = True):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock, open(self.state_dir / 'state.lock', 'a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                path = self.state_dir / 'state.json'
                try:
                    state = json.loads(path.read_text()) if path.exists() else {}
                except ValueError:
                    state = {}
                for key in ('observations', 'starts', 'logged', 'decisions', 'codex_poll'):
                    state.setdefault(key, {})
                yield state
                if write:
                    self._atomic(path, json.dumps(state, indent=2, sort_keys=True) + '\n')
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _atomic(self, path: Path, text: str) -> None:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.' + path.name, delete=False) as stream:
            stream.write(text)
        Path(stream.name).replace(path)

    def latest(self, provider: str) -> dict | None:
        with self._state(write=False) as state:
            return state['observations'].get(provider)

    # Observations

    def observe(self, observation: dict | None) -> bool:
        if not observation:
            return False
        with self._state() as state:
            provider = observation['provider']
            state['observations'][provider] = merge(state['observations'].get(provider), observation)
        return True

    def observe_events(self, provider: str, events: list, observed_at: float | None = None, source: str = '') -> bool:
        provider = provider_for(provider, self.config) or provider
        at = self.clock() if observed_at is None else observed_at
        return self.observe(observation_from_events(provider, events, at, source))

    def observe_call(self, model: str | None, destination: Path, call: dict) -> bool:
        """Read the stream of a finished provider call."""
        provider = provider_for(model, self.config)
        name = call.get('raw_stream_file') if isinstance(call, dict) else None
        if provider is None or not name or not (Path(destination) / name).is_file():
            return False
        path = Path(destination) / name
        return self.observe_events(provider, read_events(path), self.clock(), str(path))

    def scan_streams(self, roots: Iterable[Path], readings: int = 20, max_files: int = 2000) -> int:
        """Read saved provider streams under ``roots``, newest first.

        Stops after ``readings`` streams with limit data or ``max_files`` streams.
        Returns the number of streams with limit data.
        """
        files = []
        for root in roots:
            for folder, _, names in os.walk(root):
                for name in names:
                    if name.startswith('provider_attempt') and name.endswith('.stdout'):
                        path = Path(folder) / name
                        files.append((path.stat().st_mtime, path))
        found = 0
        for mtime, path in sorted(files, reverse=True)[:max_files]:
            events = read_events(path)
            for provider in ('claude', 'codex'):
                if self.observe(observation_from_events(provider, events, mtime, str(path))):
                    found += 1
            if found >= readings:
                break
        return found

    # Codex polling

    def _poll_due(self, state: dict, now: float) -> bool:
        settings = self.config['providers']['codex']
        paused = state['decisions'].get('codex', {}).get('decision') == 'pause'
        interval = settings['poll_seconds_paused'] if paused else settings['poll_seconds']
        last = state['codex_poll'].get('last_attempt_at')
        return last is None or now - last >= interval

    def refresh_codex(self, force: bool = False) -> None:
        """Poll the Codex account windows when the shared cache is old."""
        with self.poll_lock, open(self._poll_lock_path(), 'a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                now = self.clock()
                with self._state(write=False) as state:
                    if not force and not self._poll_due(state, now):
                        return
                error = None
                try:
                    result = self.codex_poller(self.config['providers']['codex']['poll_timeout_seconds'])
                    observation = codex_observation(result, now, 'codex app-server account/rateLimits/read') if result else None
                    if result and observation is None:
                        error = 'codex app-server answer had no rate-limit windows'
                except Exception as failure:  # noqa: BLE001 - a failed status request never stops a collection
                    observation, error = None, str(failure) if isinstance(failure, PollError) else repr(failure)
                with self._state() as state:
                    state['codex_poll'] = dict(last_attempt_at=now, last_error=error,
                                               last_success_at=now if observation else state['codex_poll'].get('last_success_at'))
                    if observation:
                        # A poll is authoritative for the allowed flags; windows still merge.
                        state['observations']['codex'] = merge(state['observations'].get('codex'), observation)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _poll_lock_path(self) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        return self.state_dir / 'codex_poll.lock'

    # Decisions

    def decide(self, provider: str, poll: bool = True) -> dict:
        provider = provider_for(provider, self.config) or provider
        if provider == 'codex' and poll:
            self.refresh_codex()
        now = self.clock()
        settings = self.config['providers'][provider]
        margin = self.config['resume_margin_seconds']
        observation = self.latest(provider)
        decision = dict(provider=provider, decision='run', until=None, reason='no usage observation yet',
                        windows={}, observed_at=None, source=None, stale=None)
        if not observation:
            return self._remember(decision)
        age = now - observation['observed_at']
        decision.update(observed_at=observation['observed_at'], source=observation.get('source'),
                        stale=age > settings['stale_after_seconds'], reason='below the chosen limits')
        applies = self.paced_limits(provider, observation)
        active = {}
        for name, window in observation.get('windows', {}).items():
            reset = window.get('resets_at')
            if reset is not None and now >= reset + margin:
                continue  # The window has reset since this reading.
            if provider == 'codex' and split_window_key(name)[0] not in applies:
                continue  # A bucket of another model does not limit the collection.
            active[name] = window
        decision['windows'] = active
        pauses, slows = [], []
        for name, window in active.items():
            used = window.get('used')
            if used is None:
                continue
            limits = settings['thresholds'].get(split_window_key(name)[1], settings['thresholds']['default'])
            pause_at = min(limits['pause_at'], HARD_PAUSE_AT.get(provider, 1.0))
            if used >= pause_at:
                pauses.append((f'{name} at {used:.0%} reached the pause limit {pause_at:.0%}',
                               window.get('resets_at')))
            elif used >= limits['slow_at']:
                slows.append(f'{name} at {used:.0%} reached the chosen slow limit {limits["slow_at"]:.0%}')
        reset_fallback = max((w['resets_at'] for w in active.values() if w.get('resets_at') is not None), default=None)
        if provider == 'claude':
            if observation.get('status') == 'rejected':
                reset = observation.get('rejected_resets_at')
                if reset is None or now < reset + margin:
                    pauses.append((f'provider rejected requests ({observation.get("rate_limit_type")})', reset))
            if settings.get('pause_on_extra_usage') and observation.get('is_using_overage'):
                pauses.append(('the account is using extra usage', reset_fallback))
        else:
            if observation.get('ordinary_usage_allowed') is False:
                pauses.append(('Codex reports that ordinary included usage is not allowed', reset_fallback))
            if observation.get('rate_limit_reached_type'):
                pauses.append((f'Codex reports {observation["rate_limit_reached_type"]}', reset_fallback))
            if settings.get('pause_on_extra_usage') and observation.get('spend_control_reached'):
                pauses.append(('Codex reports that spend control was reached', reset_fallback))
            for limit, info in (observation.get('buckets') or {}).items():
                if limit == CODEX_MAIN_LIMIT or limit not in applies or not info.get('rate_limit_reached_type'):
                    continue
                resets = [w['resets_at'] for key, w in active.items()
                          if split_window_key(key)[0] == limit and w.get('resets_at') is not None]
                pauses.append((f'Codex bucket {limit} reports {info["rate_limit_reached_type"]}',
                               max(resets) if resets else reset_fallback))
        if pauses:
            resets = [reset for _, reset in pauses]
            until = None if any(reset is None for reset in resets) else max(resets) + margin
            decision.update(decision='pause', until=until, reason='; '.join(reason for reason, _ in pauses))
        elif slows:
            decision.update(decision='slow', reason='; '.join(slows))
        return self._remember(decision)

    def paced_limits(self, provider: str, observation: dict) -> set:
        """Codex limit buckets that pace the collection: the main bucket and the buckets of
        the collection models (``model_slugs`` in the settings)."""
        if provider != 'codex':
            return set()
        slugs = set(self.config['providers']['codex'].get('model_slugs', []))
        return {CODEX_MAIN_LIMIT} | {limit for limit, info in (observation.get('buckets') or {}).items()
                                     if info.get('model_slug') in slugs}

    def _remember(self, decision: dict) -> dict:
        with self._state() as state:
            state['decisions'][decision['provider']] = dict(decision=decision['decision'], at=self.clock())
        return decision

    # Gate

    def gate(self, model: str) -> dict | None:
        """Wait until a new request for ``model`` may start. Unknown models pass at once."""
        provider = provider_for(model, self.config)
        if provider is None:
            return None
        settings = self.config['providers'][provider]
        recheck = self.config['recheck_seconds']
        while True:
            if self.stop.is_set():
                raise WatchStopped('User requested pause')
            decision = self.decide(provider)
            self.log(decision)
            now = self.clock()
            if decision['decision'] == 'run':
                self._claim_start(provider, now, 0)
                return decision
            if decision['decision'] == 'slow':
                wait = self._claim_start(provider, now, settings['slow_spacing_seconds'])
                if wait <= 0:
                    return decision
                self.sleep(min(wait, recheck))
                continue
            if decision['until'] is None:
                if provider == 'codex':
                    self.sleep(settings['poll_seconds_paused'])
                    continue
                # No reset time: let one request start every unknown_reset_retry_seconds. The stream of
                # that request gives a new reading (allowed, or a new rejection that may carry a reset time).
                retry = self.config.get('unknown_reset_retry_seconds', 900)
                wait = self._claim_start(provider, now, retry)
                if wait <= 0:
                    decision = dict(decision, decision='retry',
                                    reason=decision['reason'] + f'; no reset time, one retry call every {retry} s')
                    self.log(decision)
                    return decision
                self.sleep(max(0.01, min(wait, recheck)))
                continue
            self.sleep(max(0.01, min(decision['until'] - now, recheck)))

    def _claim_start(self, provider: str, now: float, spacing: float) -> float:
        """Record a request start. Return the wait when the chosen spacing is not over."""
        with self._state() as state:
            last = state['starts'].get(provider)
            if spacing and last is not None and now < last + spacing:
                return last + spacing - now
            state['starts'][provider] = now
            return 0

    # Records

    def line(self, decision: dict) -> dict:
        now = self.clock()
        return dict(at=iso(now), at_epoch=now, provider=decision['provider'], decision=decision['decision'],
                    until=iso(decision['until']), until_epoch=decision['until'], reason=decision['reason'],
                    windows={name: dict(used=w.get('used'), resets_at=iso(w.get('resets_at')))
                             for name, w in decision['windows'].items()},
                    observed_at=iso(decision['observed_at']),
                    age_seconds=None if decision['observed_at'] is None else round(now - decision['observed_at']),
                    stale=decision['stale'], source=decision['source'], pid=os.getpid(),
                    version=VERSION, config_sha256=self.config_sha256)

    def log(self, decision: dict, force: bool = False) -> bool:
        """Append a status line when the decision changes or the chosen interval passes."""
        now = self.clock()
        key = f'{decision["decision"]}|{decision["until"]}'
        with self._state() as state:
            previous = state['logged'].get(decision['provider'])
            if not force and previous and previous['key'] == key and now - previous['at'] < self.config['log_every_seconds']:
                return False
            state['logged'][decision['provider']] = dict(key=key, at=now)
            with open(self.state_dir / 'usage_watch.jsonl', 'a') as stream:
                stream.write(json.dumps(self.line(decision), sort_keys=True) + '\n')
        self.write_status_text()
        return True

    def status(self, poll: bool = True) -> dict:
        return {provider: self.decide(provider, poll=poll) for provider in self.config['providers']}

    def write_status(self, poll: bool = True) -> dict:
        decisions = self.status(poll)
        for decision in decisions.values():
            self.log(decision, force=True)
        return decisions

    def write_status_text(self) -> None:
        now = self.clock()
        with self._state(write=False) as state:
            observations = dict(state['observations'])
            decisions = dict(state['decisions'])
            poll = dict(state['codex_poll'])
        lines = [f'Usage watch at {iso(now)}. Thresholds are chosen limits in {CONFIG_PATH.name}. They are not provider values.']
        for provider in self.config['providers']:
            observation = observations.get(provider)
            decision = decisions.get(provider, {}).get('decision', 'unknown')
            parts = [f'{provider:<7}{decision:<6}']
            if observation:
                paced = self.paced_limits(provider, observation)
                for name, window in sorted(observation.get('windows', {}).items()):
                    used = window.get('used')
                    shown = '?' if used is None else f'{used:.0%}'
                    reset = window.get('resets_at')
                    note = ' (other model, not paced)' if provider == 'codex' and split_window_key(name)[0] not in paced else ''
                    if reset is not None and now >= reset + self.config['resume_margin_seconds']:
                        parts.append(f'{name} reset at {iso(reset)} (no reading since){note}')
                    else:
                        parts.append(f'{name} {shown} (resets {iso(reset) or "unknown"}){note}')
                parts.append(f'read {round((now - observation["observed_at"]) / 60)} min ago from {observation.get("source")}')
            else:
                parts.append('no usage reading yet')
            lines.append('  '.join(parts))
        if poll.get('last_error'):
            lines.append(f'Last Codex status request error: {poll["last_error"]}')
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._atomic(self.state_dir / 'status.txt', '\n'.join(lines) + '\n')


# Collector integration

def wrap_call_model(call: Callable, watcher: Callable[[], Watcher]) -> Callable:
    """Feed each finished provider stream to the watcher. Errors never reach the call."""
    def call_model(text, cwd, job, destination, attempt):
        result = call(text, cwd, job, destination, attempt)
        try:
            watcher().observe_call(job.get('model'), destination, result)
        except Exception:  # noqa: BLE001 - observation must never change a provider call
            pass
        return result
    call_model.usage_watch_wrapped = True
    return call_model


def install(central: Any, watcher: Watcher | None = None) -> Callable[[], Watcher]:
    """Patch a loaded central collector module in place and return the watcher getter.

    Before each provider request, the shared quota coordinator asks the watcher. The
    request stage and the Stata lock are unchanged: the wait happens before the collector
    records ``provider_in_flight`` and before it takes the Stata lock.
    """
    if getattr(central, 'usage_watch_holder', None) is not None:
        if watcher is not None:
            central.usage_watch_holder['watcher'] = watcher
            watcher.stop = central.STOP
        return central.usage_watcher
    holder = {'watcher': watcher}
    if watcher is not None:
        watcher.stop = central.STOP

    def get() -> Watcher:
        if holder['watcher'] is None:
            holder['watcher'] = Watcher(stop=central.STOP)
        return holder['watcher']

    base = central.QuotaCoordinator

    class WatchedQuotaCoordinator(base):
        def acquire(self, provider):
            try:
                get().gate(provider)
            except WatchStopped as error:
                raise central.Paused(str(error)) from error
            except Exception as error:  # noqa: BLE001
                # A watcher fault must not create a protocol hold. The central quota
                # handling below still applies, and the collector log keeps the fault.
                print(json.dumps(dict(usage_watch_error=repr(error), provider=provider, at=iso(time.time()))),
                      file=sys.stderr, flush=True)
            return super().acquire(provider)

    base_load = central.load_runtime

    def load_runtime(work):
        runtime = base_load(work)
        runtime.c.call_model = wrap_call_model(runtime.c.call_model, get)
        return runtime

    central.QuotaCoordinator = WatchedQuotaCoordinator
    central.load_runtime = load_runtime
    central.usage_watch_holder = holder
    central.usage_watcher = get
    return get


# Replay and command line

def replay(roots: Iterable[Path], config: dict | None = None, limit: int | None = None) -> list:
    """Decisions over saved streams in time order. No provider request."""
    files = []
    for root in roots:
        for folder, _, names in os.walk(root):
            for name in names:
                if name.startswith('provider_attempt') and name.endswith('.stdout'):
                    path = Path(folder) / name
                    files.append((path.stat().st_mtime, path))
    files = sorted(files)[-limit:] if limit else sorted(files)
    rows = []
    with tempfile.TemporaryDirectory(prefix='usage_watch_replay_') as folder:
        moment = [0.0]
        watcher = Watcher(config=config, state_dir=Path(folder), clock=lambda: moment[0],
                          codex_poller=lambda timeout: None)
        for mtime, path in files:
            moment[0] = mtime
            events = read_events(path)
            for provider in ('claude', 'codex'):
                if watcher.observe(observation_from_events(provider, events, mtime, str(path))):
                    decision = watcher.decide(provider, poll=False)
                    rows.append(dict(at=iso(mtime), path=str(path), **{k: decision[k] for k in
                                     ('provider', 'decision', 'until', 'reason', 'windows')}))
    return rows


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['status', 'replay'])
    parser.add_argument('--runs', type=Path, action='append',
                        help='Folder with saved provider streams (default for status: data/runs).')
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--no-poll', action='store_true', help='Do not start codex app-server.')
    parser.add_argument('--no-sessions', action='store_true', help='Do not read ~/.codex/sessions.')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    if args.action == 'replay':
        if not args.runs:
            parser.error('replay requires --runs')
        rows = replay(args.runs, limit=args.limit)
        counts = {}
        for row in rows:
            counts[(row['provider'], row['decision'])] = counts.get((row['provider'], row['decision']), 0) + 1
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f'{len(rows)} saved readings. Decisions: ' + ', '.join(f'{p} {d}: {n}' for (p, d), n in sorted(counts.items())))
            for row in rows:
                if row['decision'] != 'run':
                    print(f'{row["at"]} {row["provider"]} {row["decision"]}: {row["reason"]}')
        return
    watcher = Watcher(state_dir=args.state_dir)
    watcher.scan_streams(args.runs or [ROOT / 'data/runs'])
    if not args.no_sessions:
        snapshot = latest_codex_session_snapshot()
        if snapshot:
            watcher.observe(codex_observation(snapshot['payload'], snapshot['observed_at'], snapshot['source']))
    if not args.no_poll:
        watcher.refresh_codex(force=True)
    decisions = watcher.write_status(poll=False)
    if args.json:
        print(json.dumps({p: watcher.line(d) for p, d in decisions.items()}, indent=2))
    else:
        print((watcher.state_dir / 'status.txt').read_text(), end='')


if __name__ == '__main__':
    main()
