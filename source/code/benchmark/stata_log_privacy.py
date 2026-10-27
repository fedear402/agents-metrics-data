"""Remove Stata license identifiers from generated log text.

A Stata batch log starts with a banner that prints the license serial number
and the licensee. The analysis keeps its Stata logs in the repository, so the
generators call this function before they save a log.
"""
from __future__ import annotations

import re

# One license line, then any indented continuation lines of the licensee.
LICENSE_LINES = re.compile(
    r"^[ \t]*(?:Stata[ \t]+license|Serial[ \t]+number|Licensed[ \t]+to):.*\n?"
    r"(?:[ \t]+\S.*\n?)*",
    re.MULTILINE | re.IGNORECASE,
)


def strip_license_details(text: str) -> str:
    """Return ``text`` without the license, serial-number, and licensee lines."""
    return LICENSE_LINES.sub("", text)
