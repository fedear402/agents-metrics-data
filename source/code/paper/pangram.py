"""Extract the local manuscript and compare fixed inputs with Pangram."""

from __future__ import annotations

import argparse
import copy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
API = "https://text.external-api.pangram.com"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def groups(text):
    """Read consecutive balanced TeX groups, preserving nested braces."""
    result = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text) or text[pos] != "{":
            break
        start = pos + 1
        depth = 1
        pos += 1
        while depth:
            if pos >= len(text):
                raise ValueError("Unbalanced TeX group")
            if text[pos] in "{}" and (pos == 0 or text[pos - 1] != "\\"):
                depth += 1 if text[pos] == "{" else -1
            pos += 1
        result.append(text[start:pos - 1])
    return result


def pandoc(text, source, target):
    proc = subprocess.run(
        ["pandoc", "--from", source, "--to", target, "--wrap=none"],
        input=text, text=True, capture_output=True, check=True,
    )
    if proc.stderr.strip():
        raise ValueError(proc.stderr)
    return proc.stdout


def references(source):
    labels, citations = {}, {}
    for name in ("supplemental_appendix", "main_body"):
        for line in (source / "build" / f"{name}.aux").read_text().splitlines():
            if line.startswith(r"\newlabel{"):
                key, value = groups(line[len(r"\newlabel"):])
                if not key.endswith("@cref"):
                    labels[key] = groups(value)[0]
            if line.startswith(r"\bibcite{"):
                key, value = groups(line[len(r"\bibcite"):])
                fields = groups(value)
                citations[key] = (groups(fields[2])[0], fields[1])
    return labels, citations


def expand(text, labels, citations):
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    for env in ("figure", "table"):
        text = re.sub(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}", "", text, flags=re.S)
    text = re.sub(r"\\input\{tables/[^}]+\}", "", text)
    text = re.sub(r"\\documentclass(?:\[[^]]*\])?\{[^}]*\}", "", text)
    text = text.replace(r"\begin{document}", "").replace(r"\end{document}", "")

    def citation(match):
        command, keys = match.groups()
        parts = []
        last = None
        for key in keys.split(","):
            author, year = citations[key.strip()]
            if command == "citep":
                parts.append((", " if author == last else "; ") + ("" if author == last else author + ", ") + year)
            else:
                parts.append((", " if author == last else " and ") + (year if author == last else author + " (" + year + ")"))
            last = author
        result = "".join(parts)
        result = result[2:] if command == "citep" else result[5:]
        return "(" + result + ")" if command == "citep" else result

    text = re.sub(r"\\(citep|citet)\{([^}]+)\}", citation, text)

    def ref(match):
        command, keys = match.groups()
        keys = keys.split(",")
        nums = [labels[key] for key in keys]
        if command == "eqref":
            return "(" + nums[0] + ")"
        names = {"sec": "Section", "app": "Appendix", "tab": "Table", "fig": "Figure", "eq": "Equation"}
        kind = names[keys[0].split(":")[0]]
        if len(keys) == 1:
            return kind + "~" + nums[0]
        if any(key.split(":")[0] != keys[0].split(":")[0] for key in keys):
            return " and ".join(names[key.split(":")[0]] + "~" + number for key, number in zip(keys, nums))
        return ("Appendices" if kind == "Appendix" else kind + "s") + "~" + ", ".join(nums[:-1]) + " and~" + nums[-1]

    return re.sub(r"\\(Cref|eqref)\{([^}]+)\}", ref, text)


def clean_api(node):
    if isinstance(node, list):
        return [clean_api(item) for item in node if not (
            isinstance(item, dict) and (item.get("t") in {"Header", "Note"}
            or (item.get("t") == "Math" and item["c"][0]["t"] == "DisplayMath")))]
    if isinstance(node, dict):
        return {key: clean_api(value) for key, value in node.items()}
    return node


def windows(text, size=1000, step=500):
    words = list(re.finditer(r"\S+", text))
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        left, right = words[start].start(), words[end - 1].end()
        yield left, right, text[left:right]
        if end == len(words):
            break
        start += step


def prepare(out):
    source = ROOT / "paper/manuscript"
    if out.exists():
        raise ValueError("Use a new output directory to preserve existing inputs")
    files = list((source / "sections").glob("0[1-6]*.tex"))
    dependencies = [source / "numbers.tex", source / "numbers_pending.tex", source / "preamble.tex", source / "refs.bib", source / "main_body.tex"]
    dependencies += [source / "build" / f"{name}.{ext}" for name in ("main_body", "supplemental_appendix") for ext in ("aux", "pdf")]
    aux_time = (source / "build/main_body.aux").stat().st_mtime
    if any(path.stat().st_mtime > aux_time for path in files + dependencies[:4]):
        raise ValueError("Compile the manuscript first: auxiliary labels may be stale")
    out.mkdir(parents=True)
    snapshot = out / "source"
    hashes = {}
    for path in files + dependencies:
        relative = path.relative_to(source)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    labels, citations = references(snapshot)
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}", (snapshot / "numbers.tex").read_text()))
    sections, all_blocks, api_items = [], [], []
    for path in sorted(files):
        name = path.stem
        raw = (snapshot / "sections" / path.name).read_text()
        expanded = expand(raw, labels, citations)
        expanded = re.sub(r"\\(num[A-Za-z]+)(?:\{\})?", lambda m: macros[m[1]], expanded)
        def cost(match):
            command, value = match.groups()
            value = Decimal(value)
            if command == "costdollars":
                return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
            if command == "costcents":
                return format(value * 100, "f").rstrip("0").rstrip(".") if "." in format(value * 100, "f") else str(value * 100)
            return str((1 + value / 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        expanded = re.sub(r"\\(costdollars|costcents|pcttimes)\{([^{}]+)\}", cost, expanded)
        ast = json.loads(pandoc(expanded, "latex", "json"))
        serialized = json.dumps(ast)
        if '"RawInline"' in serialized or '"RawBlock"' in serialized or '"Cite"' in serialized:
            raise ValueError(f"Unresolved LaTeX in {name}")
        all_blocks.extend(ast["blocks"])
        (out / f"{name}.md").write_text(pandoc(json.dumps(ast), "json", "markdown"))
        plain = pandoc(json.dumps(clean_api(copy.deepcopy(ast))), "json", "plain").strip()
        (out / f"{name}.txt").write_text(plain + "\n")
        sections.append({"id": name, "text": plain})
    ast["blocks"] = all_blocks
    (out / "body.md").write_text(pandoc(json.dumps(ast), "json", "markdown"))
    body = "\n\n".join(section["text"] for section in sections)
    (out / "body.txt").write_text(body + "\n")
    api_items.append({"id": "whole_body", "kind": "body", "text": body})
    cursor = 0
    for section in sections:
        api_items.append({**section, "kind": "section", "start_index": cursor, "end_index": cursor + len(section["text"])})
        cursor += len(section["text"]) + 2
    for index, (left, right, text) in enumerate(windows(body), 1):
        api_items.append({"id": f"window_{index:02d}", "kind": "window", "start_index": left, "end_index": right, "text": text})
    for item in api_items:
        item["sha256"] = sha(item["text"])
        item["words"] = len(item["text"].split())
        if "start_index" in item:
            assert body[item["start_index"]:item["end_index"]] == item["text"]
    save(out / "inputs.json", api_items)
    save(out / "manifest.json", {
        "source_hashes": hashes,
        "manuscript_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "includes_local_edits": True,
        "window_words": 1000, "window_step_words": 500,
        "word_count_definition": "whitespace-delimited tokens; may differ from Pangram billing",
        "api_exclusions": ["headings", "footnotes", "display equations", "figures", "tables", "captions", "front matter", "bibliography", "appendices"],
        "markdown_exclusions": ["figures", "tables", "captions", "front matter", "bibliography", "appendices"],
        "api_docs": "https://pangram.readthedocs.io/en/latest/api/rest.html",
        "pandoc_version": subprocess.check_output(["pandoc", "--version"], text=True).splitlines()[0],
    })
    print(json.dumps({"inputs": len(api_items), "body_words": len(body.split()), "section_words": {s["id"]: len(s["text"].split()) for s in sections}}))


def validate_text(expected, returned):
    """Allow provider whitespace normalization while rejecting wording changes."""
    if not isinstance(returned, str) or " ".join(expected.split()) != " ".join(returned.split()):
        raise ValueError("Returned text differs from submitted input")
    return {"exact_match": expected == returned, "whitespace_equivalent": True,
            "submitted_sha256": sha(expected), "returned_sha256": sha(returned)}


def run(out, model="pangram-4"):
    """Resume saved tasks. Never automatically retry a POST with an unknown outcome."""
    if model != "pangram-4":
        raise ValueError("This runner requires explicit model pangram-4")
    inputs = json.loads((out / "inputs.json").read_text())
    config_path = out / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()).get("model") != model:
        raise ValueError("Saved model differs. Use a new output directory.")
    result_dir = out / "responses"
    for item in inputs:
        result_path = result_dir / (item["id"] + ".json")
        if result_path.exists():
            cached = json.loads(result_path.read_text())
            if cached.get("version") != "4.0":
                raise ValueError("Cached detector version differs from 4.0. Use a new output directory.")
            validation = validate_text(item["text"], cached.get("text"))
            save(result_path.with_suffix(".text_validation.json"), validation)
        elif (result_dir / (item["id"] + ".task.json")).exists() and not config_path.exists():
            raise ValueError("Saved task model is unknown. Use a new output directory.")
    save(config_path, {"model": model, "expected_version": "4.0"})
    credential = subprocess.run(["security", "find-generic-password", "-s", "pangram-api-token", "-a", "federicolopez", "-w"], capture_output=True)
    if credential.returncode:
        raise RuntimeError("Keychain credential unavailable")
    token = credential.stdout.decode().strip()
    class RejectedRequest(RuntimeError):
        def __init__(self, status):
            self.status = status
            super().__init__(f"Pangram HTTP {status}; no automatic retry")

    def request(route, payload=None):
        req = urllib.request.Request(API + route, data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={"x-api-key": token, "Content-Type": "application/json"})
        try:
            # 120 seconds is a chosen client timeout, not an observed API limit.
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RejectedRequest(exc.code) from None
    result_dir.mkdir(exist_ok=True)
    for item in inputs:
        if sha(item["text"]) != item["sha256"]:
            raise ValueError("Input hash mismatch")
        result_path = result_dir / (item["id"] + ".json")
        task_path = result_dir / (item["id"] + ".task.json")
        pending_path = result_dir / (item["id"] + ".pending.json")
        if result_path.exists():
            continue
        if task_path.exists():
            task = json.loads(task_path.read_text())
        else:
            if pending_path.exists():
                raise RuntimeError("An earlier POST has an unknown outcome. Inspect it before resubmission.")
            save(pending_path, {"sha256": item["sha256"], "started_at": time.time()})
            try:
                task = request("/task", {"model": model, "text": item["text"], "public_dashboard_link": False})
            except RejectedRequest as exc:
                save(result_dir / (item["id"] + ".http_error.json"), {
                    "status": exc.status, "sha256": item["sha256"], "observed_at": time.time(),
                })
                if exc.status in (400, 401, 402, 403, 413, 422, 429):
                    pending_path.unlink()
                raise
            save(task_path, task)
            pending_path.unlink()
        # Chosen polling policy: every two seconds, for up to 15 minutes per invocation.
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            response = request("/task/" + task["task_id"])
            if response.get("stage") == "STAGE_FAILED":
                save(result_path.with_suffix(".failed.json"), response)
                raise RuntimeError("Pangram task failed; inspect saved response")
            if response.get("stage") == "STAGE_SUCCESS":
                if response.get("version") != "4.0":
                    save(result_path.with_suffix(".unexpected_version.json"), response)
                    raise ValueError("Returned detector version differs from requested 4.0")
                try:
                    validation = validate_text(item["text"], response.get("text"))
                except ValueError:
                    save(result_path.with_suffix(".unexpected_text.json"), response)
                    raise
                save(result_path.with_suffix(".text_validation.json"), validation)
                save(result_path, response)
                print(json.dumps({"id": item["id"], **{key: response.get(key) for key in ("version", "prediction_short", "fraction_ai", "fraction_ai_assisted", "fraction_human")}}), flush=True)
                break
            time.sleep(2)
        else:
            raise RuntimeError("Client polling deadline reached. Run again to resume the saved task.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=["pangram-4"], default="pangram-4", help="Explicit detector model for API submissions")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out.resolve())
    else:
        run(args.out.resolve(), model=args.model)
