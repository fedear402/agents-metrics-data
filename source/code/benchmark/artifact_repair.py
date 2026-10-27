"""Model-blind, digit-preserving mechanical repair of benchmark result.json text.

Every rule only unescapes text, strips wrapper characters, or normalizes JSON
syntax. No rule may change a digit. :func:`digits_preserved` enforces that as a
hard check: the digit sequence of the original text must remain a subsequence of
the repaired text, and every inserted digit must be a zero that the number rules
place next to a decimal point.

The pipeline never closes a truncated document and never invents a missing key
or a missing object separator. A file cut off mid-write stays unrepairable by
design, because closing a partial document means guessing what the model
intended to write.

The rules carry no knowledge of the model, the arm, the task, or the reference
values. They were derived from the defect classes observed in the saved corpus,
so they describe that corpus rather than acting as a general JSON recovery tool.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable, Iterable
from typing import Any

__all__ = [
    "RULES",
    "RULE_NAMES",
    "close_unterminated_strings",
    "digits_preserved",
    "drop_trailing_commas",
    "normalize_stata_numbers",
    "null_empty_values",
    "repair_result_text",
    "strip_code_fence",
    "strip_line_wrappers",
    "strip_stray_quote_after_number",
    "undouble_quotes",
    "unescape_quotes",
]


# --- individual rules -------------------------------------------------------


def _split_ending(raw_line: str) -> tuple[str, str]:
    """Split a line into its body and its line terminator."""
    for ending in ("\r\n", "\n", "\r"):
        if raw_line.endswith(ending):
            return raw_line[: -len(ending)], ending
    return raw_line, ""


def strip_code_fence(text: str) -> str:
    """Remove a leading/trailing markdown code fence around the document."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
    stripped = re.sub(r"```\s*$", "", stripped)
    return stripped


def unescape_quotes(text: str) -> str:
    r"""Turn ``\"`` back into ``"`` (Stata/shell escaping leaked into the file)."""
    return text.replace('\\"', '"')


def undouble_quotes(text: str) -> str:
    """Turn ``""`` back into ``"`` (Stata compound-quote doubling)."""
    return text.replace('""', '"')


def strip_line_wrappers(text: str) -> str:
    """Strip Stata compound-quote wrappers around whole lines.

    Stata's ``file write ... `"..."'`` idiom leaves a stray opening quote or
    backtick-quote before a line's ``{``/``[`` and a stray ``"'`` after the
    line's closing brace. Only wrapper characters are removed.
    """
    out = []
    for raw_line in text.splitlines(keepends=True):
        line, ending = _split_ending(raw_line)
        line = re.sub(r'^(\s*)`?"(?=\s*[\{\[])', r"\1", line)
        line = re.sub(r'("\'|`")\s*$', "", line)
        line = re.sub(r'(?<=[\}\]])\s*"\s*(,?)\s*$', r"\1", line)
        out.append(line + ending)
    return "".join(out)


def close_unterminated_strings(text: str) -> str:
    """Close a string value that runs to the end of its line.

    Stata's ``file write`` sometimes drops the closing quote of a metadata
    string. A line holding an odd number of quotes gets one appended before any
    trailing structural characters. Digits are untouched.
    """
    out = []
    for raw_line in text.splitlines(keepends=True):
        line, ending = _split_ending(raw_line)
        if line.count('"') % 2 == 1:
            match = re.search(r"[\s,\}\]]*$", line)
            cut = match.start() if match else len(line)
            line = line[:cut] + '"' + line[cut:]
        out.append(line + ending)
    return "".join(out)


def drop_trailing_commas(text: str) -> str:
    """Remove a comma that immediately precedes ``}`` or ``]``."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def strip_stray_quote_after_number(text: str) -> str:
    """Strip a wrapper quote left immediately after a bare numeric value.

    The value must start right after its colon and be a complete JSON number, so
    the closing quote of a string ending in a digit (``"vcov": "HC1"``) is never
    touched.
    """
    return re.sub(
        r'(:[ \t]*-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)"(?=[ \t]*[,}\]])', r"\1", text
    )


def null_empty_values(text: str) -> str:
    """Encode an absent value as ``null`` (``"n_obs": }``); no digit is added."""
    return re.sub(r"(:[ \t]*)(?=[,}\]])", r"\1null", text)


def normalize_stata_numbers(text: str) -> str:
    """Normalize Stata numeric writers without changing any digit.

    ``.`` alone (system missing) becomes ``null``; ``.5`` becomes ``0.5``; ``5.``
    becomes ``5.0``. The first two substitutions mirror
    ``strict_posthoc.repair_stata_json``, which the canonical scorer already
    applies, so this rule only rediscovers artifacts the current scoring can
    already load.
    """
    text = re.sub(r"(?<=[:,\[\s])-?\.(?=\s*[,}\]])", "null", text)
    text = re.sub(r"(?<=[:,\[\s])(-?)\.(?=\d)", r"\g<1>0.", text)
    text = re.sub(r"(?<=\d)\.(?=\s*[,}\]])", ".0", text)
    return text


RULES: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("code_fence", strip_code_fence),
    ("escaped_quotes", unescape_quotes),
    ("doubled_quotes", undouble_quotes),
    ("line_wrapper", strip_line_wrappers),
    ("unterminated_string", close_unterminated_strings),
    ("trailing_comma", drop_trailing_commas),
    ("stray_quote_after_number", strip_stray_quote_after_number),
    ("empty_value", null_empty_values),
    ("stata_number", normalize_stata_numbers),
)

RULE_NAMES: tuple[str, ...] = tuple(name for name, _ in RULES)


# --- guards -----------------------------------------------------------------


def digits_preserved(raw: str, repaired: str) -> bool:
    """True when no digit was changed: raw digits are a subsequence of repaired.

    Any digit present only in the repaired text must be a zero, the padding the
    number rules insert next to a decimal point.
    """
    original = [c for c in raw if c.isdigit()]
    index = 0
    for char in repaired:
        if not char.isdigit():
            continue
        if index < len(original) and char == original[index]:
            index += 1
        elif char != "0":
            return False
    return index == len(original)


def _normalize_empty_containers(data: dict[str, Any]) -> dict[str, Any]:
    """Mirror ``strict_posthoc.load_result``: jsonlite writes ``[]`` for ``{}``."""
    for key in ("diagnostics", "method_metadata"):
        if data.get(key) == []:
            data[key] = {}
    return data


# --- driver -----------------------------------------------------------------


def _unrepairable_label(raw: str) -> str:
    """Sub-label an unrepairable file so truncation stays visible in the tables."""
    body = raw.strip()
    if not body:
        return "unrepairable: empty file"
    if not body.endswith("}"):
        return "unrepairable: truncated"
    return "unrepairable: other"


def _apply(raw: str, names: Iterable[str]) -> str:
    wanted = set(names)
    text = raw
    for name, rule in RULES:
        if name in wanted:
            text = rule(text)
    return text


def _parses(text: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(text)
    except Exception:  # noqa: BLE001 - a bad parse is the normal negative case
        return None
    return loaded if isinstance(loaded, dict) else None


def repair_result_text(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Parse ``raw`` under the smallest set of rules that makes it valid JSON.

    Returns the parsed object and a label: ``none needed`` when the text was
    already valid JSON, a ``+``-joined minimal rule set, ``digit_guard_violation``
    when the only parsing repair would have changed a digit, or an
    ``unrepairable: ...`` label. The rule set is minimized greedily over subsets
    in increasing size, so a rule that only touched whitespace is never credited
    with the repair.
    """
    if not raw.strip():
        return None, "unrepairable: empty file"
    loaded = _parses(raw)
    if loaded is not None:
        return _normalize_empty_containers(loaded), "none needed"

    for size in range(1, len(RULE_NAMES) + 1):
        for subset in itertools.combinations(RULE_NAMES, size):
            text = _apply(raw, subset)
            loaded = _parses(text)
            if loaded is None:
                continue
            if not digits_preserved(raw, text):
                return None, "digit_guard_violation"
            return _normalize_empty_containers(loaded), "+".join(subset)
    return None, _unrepairable_label(raw)
