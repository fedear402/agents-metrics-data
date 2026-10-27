"""Export manuscript sections and a combined paper as readable plain text."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from pangram import clean_api, expand, references

ROOT = Path(__file__).resolve().parents[2]
HEADING = re.compile(r"\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\s*(?:\[[^\]]*\])?\s*\{")
LEVEL = {name: i for i, name in enumerate(("section", "subsection", "subsubsection", "paragraph", "subparagraph"), 1)}
INCLUDE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}")
VERBATIM = re.compile(r"\\begin\{(promptbox|verbatim\*?|lstlisting)\}.*?\\end\{\1\}", re.S)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return path.read_bytes().decode("utf-8")


def mask(text):
    """Hide comments and literal environments without changing offsets."""
    text = VERBATIM.sub(lambda m: " " * len(m[0]), text)
    return re.sub(r"(?<!\\)%[^\n]*", lambda m: " " * len(m[0]), text)


def group_end(text, start):
    depth = 1
    pos = start
    while pos < len(text):
        if text[pos] in "{}" and (pos == 0 or text[pos - 1] != "\\"):
            depth += 1 if text[pos] == "{" else -1
        if depth == 0:
            return pos
        pos += 1
    raise ValueError("Unbalanced heading braces")


def headings(text):
    visible = mask(text)
    found = []
    for match in HEADING.finditer(visible):
        end = group_end(visible, match.end())
        found.append({"start": match.start(), "level": LEVEL[match[1]], "title": text[match.end():end]})
    for i, item in enumerate(found):
        item["end"] = next((other["start"] for other in found[i + 1:] if other["level"] <= item["level"]), len(text))
    return found


def slug(title):
    title = re.sub(r"\\[A-Za-z]+", "", title)
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def resolve(source, relative, parent=None):
    candidate = Path(relative)
    if not candidate.suffix:
        candidate = candidate.with_suffix(".tex")
    options = [source / candidate]
    if parent is not None:
        options.append(parent / candidate)
    for path in options:
        if path.is_file():
            path = path.resolve()
            path.relative_to(source)
            return path
    raise FileNotFoundError(relative)


def flatten(text, source, dependencies, parent=None, stack=()):
    """Expand included files for Markdown only. Keep source exports untouched."""
    visible = mask(text)
    result, cursor = [], 0
    for match in INCLUDE.finditer(visible):
        path = resolve(source, match[1], parent)
        if path in stack:
            raise ValueError(f"Include cycle: {path}")
        dependencies.add(path)
        result.append(text[cursor:match.start()])
        result.append(flatten(read(path), source, dependencies, path.parent, (*stack, path)))
        cursor = match.end()
    return "".join(result) + text[cursor:]


def pandoc(text, source_format, target):
    proc = subprocess.run(["pandoc", "-f", source_format, "-t", target, "--wrap=none"],
                          input=text, capture_output=True, text=True, check=True)
    if proc.stderr.strip():
        raise RuntimeError(proc.stderr)
    return proc.stdout


def commands(text, name, count, replacement):
    pattern = re.compile(r"\\" + name + r"(?![A-Za-z])\s*")
    cursor = 0
    while match := pattern.search(text, cursor):
        pos = match.end()
        args = []
        for _ in range(count):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text) or text[pos] != "{":
                raise ValueError(f"Expected argument for {name}")
            end = group_end(text, pos + 1)
            args.append(text[pos + 1:end])
            pos = end + 1
        value = replacement(args)
        text = text[:match.start()] + value + text[pos:]
        cursor = match.start() + len(value)
    return text


def plain_document(ast):
    """Render content without adding Markdown delimiters, fences, or table rules."""
    def inline(nodes):
        result = []
        for node in nodes:
            kind, content = node["t"], node.get("c")
            if kind == "Str":
                value = content
            elif kind in {"Space", "SoftBreak"}:
                value = " "
            elif kind == "LineBreak":
                value = "\n"
            elif kind in {"Emph", "Strong", "Underline", "Strikeout", "SmallCaps"}:
                value = inline(content)
            elif kind in {"Superscript", "Subscript"}:
                value = ("^(" if kind == "Superscript" else "_(") + inline(content) + ")"
            elif kind == "Quoted":
                quote = '"' if content[0]["t"] == "DoubleQuote" else "'"
                value = quote + inline(content[1]) + quote
            elif kind in {"Span", "Cite"}:
                value = inline(content[1])
            elif kind == "Code":
                value = content[1]
            elif kind == "Math":
                fragment = {**ast, "meta": {}, "blocks": [{"t": "Plain", "c": [node]}]}
                value = pandoc(json.dumps(fragment), "json", "plain").strip()
            elif kind == "Note":
                value = " (Footnote: " + blocks(content).replace("\n\n", " ") + ")"
            elif kind == "Link":
                label, target = inline(content[1]), content[2][0]
                value = label if label == target else label + " (" + target + ")"
            elif kind == "Image":
                label = inline(content[1])
                value = (label + "\n" if label else "") + "Figure file: " + content[2][0]
            else:
                raise ValueError(f"Unsupported plain-text inline: {kind}")
            result.append(value)
        return "".join(result)

    def rows(values):
        return ["\t".join(blocks(cell[-1]).replace("\n\n", " ") for cell in row[-1]) for row in values]

    def blocks(nodes):
        result = []
        for node in nodes:
            kind, content = node["t"], node.get("c")
            if kind in {"Para", "Plain"}:
                value = inline(content)
            elif kind == "Header":
                value = inline(content[2])
            elif kind == "CodeBlock":
                value = content[1]
            elif kind in {"BlockQuote", "Div"}:
                value = blocks(content if kind == "BlockQuote" else content[1])
            elif kind == "LineBlock":
                value = "\n".join(inline(line) for line in content)
            elif kind == "BulletList":
                value = "\n".join(blocks(item) for item in content)
            elif kind == "OrderedList":
                value = "\n".join(f"Item {i}: " + blocks(item) for i, item in enumerate(content[1], content[0][0]))
            elif kind == "DefinitionList":
                value = "\n".join(inline(term) + ": " + " ".join(blocks(d) for d in definitions) for term, definitions in content)
            elif kind == "Table":
                _, caption, _, head, bodies, foot = content
                values = [blocks(caption[1]), *rows(head[1])]
                for body in bodies:
                    values.extend(rows(body[2]))
                    values.extend(rows(body[3]))
                values.extend(rows(foot[1]))
                value = "\n".join(v for v in values if v)
            elif kind == "Figure":
                value = blocks(content[1][1]) + "\n" + blocks(content[2])
            elif kind in {"HorizontalRule", "Null"}:
                value = ""
            else:
                raise ValueError(f"Unsupported plain-text block: {kind}")
            if value:
                result.append(value)
        return "\n\n".join(result)
    return blocks(ast["blocks"]).strip() + "\n"


def markdown(text, source, dependencies, labels, citations, prose_only=False):
    if prose_only:
        for env in ("figure", "table"):
            text = re.sub(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}", "", text, flags=re.S)
        text = re.sub(r"\\input\{tables/[^}]+\}", "", text)
    text = flatten(text, source, dependencies)
    literals = []
    def protect(match):
        literals.append(match[0])
        return f"LITERALPLACEHOLDER{len(literals) - 1}END"
    text = VERBATIM.sub(protect, text)
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    text = re.sub(r"\\documentclass(?:\[[^]]*\])?\{[^}]*\}", "", text)
    text = text.replace(r"\begin{document}", "").replace(r"\end{document}", "")
    text = text.replace(r"\begin{abstract}", r"\section*{Abstract}").replace(r"\end{abstract}", "")
    # Resolve only citations and references through the earlier extractor.
    # Its exhibit-removal rules never receive any other manuscript content.
    text = re.sub(r"\\(?:citep|citet|Cref|eqref)\{[^}]+\}", lambda m: expand(m[0], labels, citations), text)
    numeric_source = read(source / "numbers.tex")
    macros = dict(re.findall(r"\\(?:newcommand|providecommand)\{\\(\w+)\}\{([^{}]*)\}", numeric_source))
    text = re.sub(r"\\(num[A-Za-z]+)(?:\{\})?", lambda m: macros[m[1]], text)
    def numeric(match):
        command, places, value = match.groups()
        value = Decimal(value)
        if command == "costdollars":
            return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        if command == "costcents":
            cents = value * 100
            return format(cents.quantize(Decimal(10) ** -int(places), rounding=ROUND_HALF_UP), "f") if places else format(cents.normalize(), "f")
        return str((1 + value / 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    text = re.sub(r"\\(costdollars|costcents|pcttimes)(?:\[(\d+)\])?\{([^{}]+)\}", numeric, text)
    text = commands(text, "raminutes", 2, lambda a: str((60 * Decimal(a[0]).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) / Decimal(a[1])).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)))
    for name, count in (("setlength", 2), ("setstretch", 1), ("phantom", 1), ("shortTitle", 1), ("needspace", 1), ("Needspace", 1)):
        text = commands(text, name, count, lambda a: "")
    text = re.sub(r"\\(?:centering|raggedright|arraybackslash|footnotesize|small|scriptsize|AddSubmissionCountMark|maketitle)(?![A-Za-z])", "", text)
    text = re.sub(r"\\(?:addlinespace|vspace)\*?(?:\[[^]]*\]|\{[^}]*\})?", "", text)
    text = re.sub(r"\\(?:quad|qquad|FloatBarrier)(?![A-Za-z])|\\allowbreak(?:\{\})?", "", text)
    text = re.sub(r"\\hangindent\s*=\s*[\d.]+[a-z]+|\\hangafter\s*=\s*\d+", "", text)
    text = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", text)
    text = commands(text, "mbox", 1, lambda a: a[0])
    text = re.sub(r"\\(?:begin|end)\{(?:footnotesize|small|scriptsize|flushleft|center)\}", "", text)
    text = re.sub(r"\\begin\{minipage\}(?:\[[^]]*\])?\{[^}]*\}|\\end\{minipage\}", "", text)
    text = re.sub(r"\\begin\{tabular\*\}\{[^}]*\}", r"\\begin{tabular}", text).replace(r"\end{tabular*}", r"\end{tabular}")
    text = text.replace(r"\\*", r"\\")
    text = commands(text, "multicolumn", 3, lambda a: r"\multicolumn{" + a[0] + "}{l}{" + a[2] + "}" + (r" \\" if a[2].startswith(r"\textit{Notes:}") else ""))
    # Harvard bibliography labels are metadata. The following entry text is printed.
    text = re.sub(r"\\harvarditem\[[^]]*\]", r"\\harvarditem", text)
    text = commands(text, "harvarditem", 3, lambda a: "\n\n")
    text = text.replace(r"\harvardand", "and")
    text = re.sub(r"\\begin\{thebibliography\}\{[^}]*\}", r"\\section*{References}", text).replace(r"\end{thebibliography}", "")
    for i, literal in enumerate(literals):
        literal = literal.replace(r"\begin{promptbox}", r"\begin{verbatim}").replace(r"\end{promptbox}", r"\end{verbatim}")
        text = text.replace(f"LITERALPLACEHOLDER{i}END", literal)
    ast = json.loads(pandoc(text, "latex+raw_tex", "json"))
    metadata = []
    for key in ("title", "author", "date"):
        value = ast.get("meta", {}).get(key)
        if not value:
            continue
        entries = value["c"] if value["t"] == "MetaList" else [value]
        for entry in entries:
            if entry["t"] == "MetaInlines":
                metadata.append({"t": "Para", "c": entry["c"]})
    ast["blocks"] = metadata + ast["blocks"]
    ast["meta"] = {}
    preserved = []
    def walk(node):
        if isinstance(node, list):
            return [walk(value) for value in node]
        if not isinstance(node, dict):
            return node
        if node.get("t") in {"RawInline", "RawBlock"}:
            raw = node["c"][1]
            if re.fullmatch(r"\\label\{[^}]*\}", raw):
                return {"t": "Str", "c": ""} if node["t"] == "RawInline" else {"t": "Null"}
            preserved.append(raw)
            return {"t": "Code" if node["t"] == "RawInline" else "CodeBlock", "c": [["", ["latex"], []], raw]}
        if node.get("t") == "Image":
            target = node["c"][2][0]
            path = resolve(source, target)
            dependencies.add(path)
            node["c"][2][0] = "_sources/" + path.relative_to(source).as_posix()
        return {key: walk(value) for key, value in node.items()}
    ast = walk(ast)
    if prose_only:
        ast = clean_api(ast)
    return plain_document(ast), preserved


def export(source, out):
    source = source.resolve()
    out = out.resolve()
    labels, citations = references(source)
    dependencies = {source / name for name in ("main.tex", "preamble.tex", "numbers.tex", "numbers_pending.tex", "refs.bib", "AEA.cls", "aea.bst")}
    dependencies.update(source / "build" / f"{name}.aux" for name in ("main_body", "supplemental_appendix"))
    units = []
    def add(name, title, path, start, end, level):
        units.append(dict(name=name, title=title, source=str(path.relative_to(source)), start=start, end=end, level=level, raw=read(path)[start:end]))
    front = source / "sections/00_frontmatter.tex"
    dependencies.add(front)
    front_text = read(front)
    abstract = re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", mask(front_text), re.S)
    if abstract is None:
        raise ValueError("Abstract environment missing")
    add("abstract", "Abstract", front, abstract.start(), abstract.end(), 1)
    add("frontmatter", "Front matter", front, 0, len(front_text), 0)
    main = read(source / "main.tex")
    # Front matter already contains the abstract. Parent sections contain all
    # their subsections, so only complete source files enter the combined paper.
    whole_order = [(-1, "frontmatter")]
    for match in re.finditer(r"\\subfile\{([^}]+)\}", mask(main)):
        path = resolve(source, match[1])
        dependencies.add(path)
        text = read(path)
        found = headings(text)
        title = found[0]["title"] if found else "Introduction"
        prefix = "appendix_" + path.stem.split("_")[0].lower() + "_" if path.parent.name == "appendix" else ""
        name = prefix + slug(title)
        add(name, title, path, 0, len(text), 1)
        whole_order.append((match.start(), name))
        document_end = text.rfind(r"\end{document}")
        for i, section in enumerate(found):
            if i == 0 and section["level"] == 1:
                continue
            parents = [h for h in found[:i] if h["start"] < section["start"] < h["end"] and h["level"] < section["level"]]
            child = prefix + "__".join(slug(h["title"]) for h in [*parents, section])
            add(child, section["title"], path, section["start"], min(section["end"], document_end if document_end >= 0 else len(text)), section["level"])
    # The compiled bibliography is the exact LaTeX that prints the references.
    bibliography = source / "build/main_body.bbl"
    dependencies.add(bibliography)
    add("references", "References", bibliography, 0, len(read(bibliography)), 1)
    bibliography_command = re.search(r"\\bibliography\s*\{[^}]+\}", mask(main))
    whole_order.append((bibliography_command.start() if bibliography_command else len(main), "references"))
    names = [unit["name"] for unit in units]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate export filenames")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="section-export-", dir=out.parent) as temporary:
        stage = Path(temporary)
        records = []
        for unit in units:
            raw = unit.pop("raw")
            md, preserved = markdown(raw, source, dependencies, labels, citations)
            (stage / (unit["name"] + ".tex")).write_bytes(raw.encode())
            (stage / (unit["name"] + ".md")).write_text(md)
            records.append({**unit, "tex_sha256": digest(raw.encode()), "md_sha256": digest(md.encode()), "preserved_latex": preserved})
        whole_sections = [name for _, name in sorted(whole_order)]
        whole_text = "\n\n".join((stage / f"{name}.md").read_text().strip() for name in whole_sections) + "\n"
        (stage / "whole_paper.md").write_text(whole_text)
        whole_paper = {"file": "whole_paper.md", "sections": whole_sections, "md_sha256": digest(whole_text.encode())}
        # Mirror source paths beside the raw exports, so includes remain recoverable.
        pending = list(dependencies)
        while pending:
            path = pending.pop()
            if path.suffix != ".tex":
                continue
            for match in INCLUDE.finditer(mask(read(path))):
                included = resolve(source, match[1], path.parent)
                if included not in dependencies:
                    dependencies.add(included)
                    pending.append(included)
        sources = {}
        for path in sorted(dependencies):
            relative = path.relative_to(source)
            target = stage / "_sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            sources[str(relative)] = digest(path.read_bytes())
        manifest = {"format_version": 2, "md_content_format": "plain text", "offset_unit": "Unicode characters", "sections": records, "whole_paper": whole_paper, "source_hashes": sources}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        index = "Manuscript by section\n\nGenerated by code/paper/export_sections.py after local compilation.\n\nLaTeX exports are exact source slices. Parent sections include their subsections. The .md files contain plain text, with no added Markdown markup. They retain equations, tab-separated table cells, captions, footnotes, and literal code. Symbols that belong to code or mathematics remain. The _sources/ folder preserves included files and figures. Unsupported LaTeX remains explicit and is listed in manifest.json. Figure file paths refer to copied assets. Separate LaTeX content files may require the original preamble to compile.\n\n"
        index += "Whole paper: whole_paper.md\n\nThe combined file contains front matter, including the abstract, followed by complete sections and references in manuscript order. Parent files supply their subsections once. The manifest records the component order and combined-file hash.\n\n"
        index += "\n".join(f"{u['title']}: {u['name']}.md, {u['name']}.tex" for u in records) + "\n"
        (stage / "README.md").write_text(index)
        if out.exists():
            if not (out / "manifest.json").exists():
                raise ValueError("Refusing to replace an unmanaged output folder")
            old = json.loads((out / "manifest.json").read_text())
            managed = {"README.md", "manifest.json", "_sources"}
            managed.update(u["name"] + ext for u in old["sections"] for ext in (".tex", ".md"))
            if old.get("whole_paper", {}).get("file") == "whole_paper.md":
                managed.add("whole_paper.md")
            if any(path.name not in managed for path in out.iterdir()):
                raise ValueError("Output folder contains unmanaged files")
            shutil.rmtree(out)
        shutil.copytree(stage, out)
    print(f"Section exports: {len(records)} LaTeX/Markdown pairs and whole_paper.md in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "paper/manuscript")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    export(args.source, args.out or args.source / "by_section")
