"""Prepare plain-text section, window, and whole-text inputs for Pangram 4."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from export_sections import markdown, references
from pangram import ROOT, save, sha, windows


def prepare(out, scope="main"):
    exports = ROOT / "paper/manuscript/by_section"
    manifest = json.loads((exports / "manifest.json").read_text())
    if manifest.get("md_content_format") != "plain text":
        raise ValueError("Compile the manuscript with plain-text exports first")
    if out.exists():
        raise ValueError("Use a new output directory")
    for name, checksum in manifest["source_hashes"].items():
        if hashlib.sha256((ROOT / "paper/manuscript" / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Source changed since compilation: " + name)
    out.mkdir(parents=True)
    shutil.copytree(exports, out / "source")
    source = out / "source/_sources"
    labels, citations = references(source)
    selected = [unit for unit in manifest["sections"] if unit["name"] not in {"frontmatter", "references"}
                and (scope == "all" or (unit["level"] == 1 and not unit["name"].startswith("appendix_")))]
    items = []
    for unit in selected:
        raw = (out / "source" / (unit["name"] + ".tex")).read_text()
        text, unsupported = markdown(raw, source, set(), labels, citations, prose_only=True)
        if unsupported:
            raise ValueError("Unresolved LaTeX in " + unit["name"])
        text = text.strip()
        (out / (unit["name"] + ".txt")).write_text(text + "\n")
        items.append({"id": unit["name"], "kind": "section", "level": unit["level"], "text": text})
    parents = [item for item in items if item["level"] == 1]
    whole = "\n\n".join(item["text"] for item in parents)
    cursor = 0
    for item in parents:
        item.update(start_index=cursor, end_index=cursor + len(item["text"]))
        cursor += len(item["text"]) + 2
    for i, (start, end, text) in enumerate(windows(whole), 1):
        items.append({"id": f"window_{i:02d}", "kind": "window", "text": text, "start_index": start, "end_index": end})
    items.append({"id": "whole_text", "kind": "whole", "text": whole})
    for item in items:
        item.update(sha256=sha(item["text"]), words=len(item["text"].split()))
        if "start_index" in item:
            assert whole[item["start_index"]:item["end_index"]] == item["text"]
    (out / "whole_text.txt").write_text(whole + "\n")
    save(out / "inputs.json", items)
    save(out / "manifest.json", {
        "scope": scope, "detector_model": "pangram-4", "expected_version": "4.0",
        "source_hashes": manifest["source_hashes"], "sequence": "sections, windows, whole text",
        "whole_text_includes": [item["id"] for item in parents],
        "window_words": 1000, "step_words": 500,
        "exclusions": ["headings", "footnotes", "display equations", "figures", "tables", "captions", "bibliography", "author front matter"],
        "word_count": "Whitespace-delimited tokens. Pangram uses its own counter.",
        "price_basis": "User dashboard: Pangram 4 realtime costs USD 0.05 per 100 words. Actual billable word counts may differ.",
    })
    estimate = sum((item["words"] + 99) // 100 for item in items) * 0.05
    print(json.dumps({"inputs": len(items), "sections": len(selected), "whole_words": len(whole.split()), "submitted_words": sum(item["words"] for item in items), "estimated_cost_usd": round(estimate, 2)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", choices=["main", "all"], default="main")
    args = parser.parse_args()
    prepare(args.out.resolve(), args.scope)
