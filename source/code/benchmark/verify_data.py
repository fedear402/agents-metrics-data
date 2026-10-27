"""Verify the complete research data against the committed SHA-256 inventory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOTS = ("data/runs", "data/analysis", "data/tasks")
REPO = Path(__file__).resolve().parents[2]


def paths(root):
    for relative in ROOTS:
        start = root / relative
        if not start.is_dir():
            raise ValueError(f"Missing data directory: {relative}")
        for directory, dirs, files in os.walk(start, followlinks=False):
            dirs.sort()
            for name in sorted(dirs + files):
                path = Path(directory) / name
                if path.is_symlink() or path.is_file():
                    yield path


def record(path, root):
    row = {"path": path.relative_to(root).as_posix()}
    before = path.lstat()
    if path.is_symlink():
        row.update(type="symlink", target=os.readlink(path))
    else:
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").hexdigest()
        row.update(type="file", bytes=before.st_size, sha256=value)
    after = path.lstat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during verification: {row['path']}")
    return row


def write_inventory(root, inventory):
    root, inventory = Path(root).resolve(), Path(inventory)
    with inventory.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as out:
            for path in paths(root):
                out.write(
                    (json.dumps(record(path, root), sort_keys=True) + "\n").encode()
                )


def verify(root, inventory):
    root = Path(root).resolve()
    with gzip.open(inventory, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    expected = {row["path"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate path in data inventory")
    seen, byte_count, regular_count, link_count = set(), 0, 0, 0
    for path in paths(root):
        row = record(path, root)
        name = row["path"]
        if name not in expected:
            raise ValueError(f"Unrecorded data file: {name}")
        if row != expected[name]:
            raise ValueError(f"Data differs from inventory: {name}")
        seen.add(name)
        byte_count += row.get("bytes", 0)
        regular_count += row["type"] == "file"
        link_count += row["type"] == "symlink"
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"Missing {len(missing)} data paths; first: {min(missing)}")
    result = {
        "verified_files": regular_count,
        "verified_links": link_count,
        "bytes": byte_count,
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--inventory", type=Path, default=REPO / "data/INVENTORY.jsonl.gz"
    )
    parser.add_argument(
        "--write-inventory",
        action="store_true",
        help="Record an intentional data update instead of verifying it.",
    )
    args = parser.parse_args()
    if args.write_inventory:
        write_inventory(args.root, args.inventory)
    else:
        verify(args.root, args.inventory)


if __name__ == "__main__":
    main()
