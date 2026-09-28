"""Resolve the thesis content into a numbered document model for the Word builder.

Steps: parse content/*.txt -> number chapters, sections, figures, tables, equations ->
assign IEEE citation numbers by first appearance -> resolve cross references ->
render table PNGs (with citation numbers) and equation PNGs -> write
data/document_model.json and the latex_support files.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from equations import EQUATIONS  # noqa: E402
from refs import REFS, bibtex, ieee  # noqa: E402
from figstyle import pretty  # noqa: E402

import make_tables  # noqa: E402

ROOT = HERE.parent
CONTENT = HERE / "content"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
EQPNG = HERE / "eq_png"
DATA = ROOT / "data"
LATEX = ROOT / "latex_support"

PARTS = [
    ("00_front_matter", "Front Matter"),
    ("01_introduction", "Chapter 1 - Introduction"),
    ("02_literature_review", "Chapter 2 - Literature Review"),
    ("03_requirements", "Chapter 3 - Requirements, Impacts and Constraints"),
    ("04_methodology", "Chapter 4 - Proposed Methodology"),
    ("05_implementation", "Chapter 5 - Implementation and Experimental Setup"),
    ("06_results", "Chapter 6 - Results and Discussion"),
    ("07_conclusion", "Chapter 7 - Conclusion and Future Work"),
    ("08_appendices", "Appendices"),
]

META = {
    "title": "RAHC-LoRA: Reward-Aware Hessian-Free Consolidation for Continual Reinforcement Learning in LLMs",
    "authors": [
        ("Ayhan Arash Tasin", "24241122"),
        ("Afiah Tabassum Mou", "22299142"),
        ("Pushpita Gosh", "22299024"),
        ("Fauzia Abida Raida", "23301149"),
        ("Nihad Hasan Niloy", "24241124"),
    ],
    "department": "Department of Computer Science and Engineering",
    "university": "BRAC University",
    "degree": "B.Sc. in Computer Science and Engineering",
    "date": "[Month] 2026",
    "semester": "[Semester], 2026",
    "approval_date": "[Date]",
    "supervisor": ("Mr. Md. Tanzim Reza", "Senior Lecturer"),
    "coordinator": ("Dr. Md. Golam Rabiul Alam", "Professor"),
    "head": ("Sadia Hamid Kazi, PhD", "Chairperson and Associate Professor"),
}

NOMENCLATURE = [
    ("A[i, j]", "Held-out score on task j after training on task i"),
    ("ADR", "Architecture Decision Record"),
    ("AdamW", "Adam optimizer with decoupled weight decay"),
    ("ARC", "AI2 Reasoning Challenge (ARC-Challenge, ARC-Easy)"),
    ("BF16 / FP32", "Brain floating point 16-bit / IEEE 32-bit floating point"),
    ("BWT", "Backward Transfer"),
    ("CF", "Catastrophic Forgetting"),
    ("CI", "Confidence Interval"),
    ("CL", "Continual Learning"),
    ("CPO", "Continual Policy Optimization"),
    ("EMA", "Exponential Moving Average"),
    ("EWC", "Elastic Weight Consolidation"),
    ("GEM / A-GEM", "(Averaged) Gradient Episodic Memory"),
    ("GPU / VRAM", "Graphics Processing Unit / Video memory"),
    ("GRPO", "Group Relative Policy Optimization"),
    ("GSM8K", "Grade School Math 8K benchmark"),
    ("HLoRA / HLER", "Hierarchical Layer-wise and Element-wise Regularization"),
    ("HLoRA-RL", "Direct transfer of HLoRA to RL (Phase 3 baseline)"),
    ("IFEval", "Instruction-Following Evaluation benchmark"),
    ("KL", "Kullback-Leibler divergence"),
    ("LLM", "Large Language Model"),
    ("LoRA", "Low-Rank Adaptation"),
    ("MAS", "Memory Aware Synapses"),
    ("MBPP", "Mostly Basic Python Programming benchmark"),
    ("MMLU", "Massive Multitask Language Understanding benchmark"),
    ("PEFT", "Parameter-Efficient Fine-Tuning"),
    ("PPL", "Perplexity"),
    ("PPO", "Proximal Policy Optimization"),
    ("pp", "Percentage points"),
    ("RAHC-LoRA", "Reward-Aware Hessian-Free Consolidation for LoRA (proposed method)"),
    ("RFT", "Reinforcement Fine-Tuning"),
    ("RL", "Reinforcement Learning"),
    ("RLHF", "Reinforcement Learning from Human Feedback"),
    ("RLVR", "Reinforcement Learning with Verifiable Rewards"),
    ("SFT", "Supervised Fine-Tuning"),
    ("SHA-256", "Secure Hash Algorithm, 256-bit"),
    ("SI", "Synaptic Intelligence"),
    ("ΔW", "Effective LoRA update (α/r)BA"),
    ("λ_f, λ_p", "Functional (KL) multiplier and parameter-penalty weight"),
]

INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|_\{[^}]*\}|\^\{[^}]*\})")
CITE_RE = re.compile(r"\[(@[A-Za-z0-9_]+(?:\s*;\s*@[A-Za-z0-9_]+)*)\]")
XREF_RE = re.compile(r"\{(fig|tab|eq|sec|chap|app):([A-Za-z0-9_]+)\}")
LABEL_RE = re.compile(r"\s*\{#([a-z]+:[A-Za-z0-9_]+)\}\s*$")


def runs(text: str) -> list[dict]:
    """Split inline markup into docx runs."""
    out = []
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            out.append({"text": text[pos:m.start()]})
        tok = m.group(0)
        if tok.startswith("**"):
            out.append({"text": tok[2:-2], "bold": True})
        elif tok.startswith("`"):
            out.append({"text": tok[1:-1], "code": True})
        elif tok.startswith("_{"):
            out.append({"text": tok[2:-1], "sub": True})
        elif tok.startswith("^{"):
            out.append({"text": tok[2:-1], "sup": True})
        else:
            out.append({"text": tok[1:-1], "italic": True})
        pos = m.end()
    if pos < len(text):
        out.append({"text": text[pos:]})
    return [r for r in out if r["text"]]


def parse(part: str) -> list[dict]:
    lines = (CONTENT / f"{part}.txt").read_text(encoding="utf-8").split("\n")
    blocks: list[dict] = []
    para: list[str] = []
    i = 0

    def flush():
        if para:
            blocks.append({"type": "para", "raw": " ".join(s.strip() for s in para)})
            para.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
            continue
        if stripped.startswith("[[CODE:"):
            flush()
            title = stripped[len("[[CODE:"):-2]
            body = []
            i += 1
            while lines[i].strip() != "[[/CODE]]":
                body.append(lines[i].rstrip())
                i += 1
            blocks.append({"type": "code", "title": title, "lines": body})
            i += 1
            continue
        m = re.match(r"^\[\[(FIG|TAB|EQ):([A-Za-z0-9_]+)\]\]\s*(.*)$", stripped)
        if m:
            flush()
            kind, ident, cap = m.groups()
            blocks.append({"type": kind.lower(), "id": ident, "caption_raw": cap})
            i += 1
            continue
        m = re.match(r"^\[\[([A-Z]+)\]\]$", stripped)
        if m:
            flush()
            blocks.append({"type": "special", "name": m.group(1)})
            i += 1
            continue
        m = re.match(r"^(#\*|#A|####|###|##|#)\s+(.*)$", stripped)
        if m:
            flush()
            marks, title = m.groups()
            label = None
            lm = LABEL_RE.search(title)
            if lm:
                label = lm.group(1)
                title = title[: lm.start()].rstrip()
            level = {"#": 1, "#*": 1, "#A": 1, "##": 2, "###": 3, "####": 4}[marks]
            blocks.append({"type": "heading", "level": level, "title": title, "label": label,
                           "kind": {"#": "chapter", "#*": "unnumbered", "#A": "appendix"}.get(marks, "section")})
            i += 1
            continue
        if stripped.startswith("- "):
            flush()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            blocks.append({"type": "bullets", "raw_items": items})
            continue
        if re.match(r"^\d+\.\s", stripped):
            flush()
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            blocks.append({"type": "numbered", "raw_items": items})
            continue
        if stripped.startswith("> "):
            flush()
            blocks.append({"type": "quote", "raw": stripped[2:]})
            i += 1
            continue
        para.append(stripped)
        i += 1
    flush()
    return blocks


def main() -> None:
    model_parts = []
    labels: dict[str, str] = {}
    figures, tables, equations = [], [], []
    cite_order: list[str] = []
    problems: list[str] = []
    tables_spec = make_tables.TABLES

    def note_cites(text: str):
        for m in CITE_RE.finditer(text):
            for key in [k.strip().lstrip("@") for k in m.group(1).split(";")]:
                if key not in REFS:
                    problems.append(f"unknown citation key {key}")
                elif key not in cite_order:
                    cite_order.append(key)
        for m in re.finditer(r"@([A-Za-z0-9_]+)", text):
            key = m.group(1)
            if key in REFS and key not in cite_order and f"@{key}" in text and "[" not in text:
                cite_order.append(key)

    # ---------- pass 1: numbering and citation order
    chapter_no = 0
    appendix_no = 0
    parsed = {}
    for part, _ in PARTS:
        blocks = parse(part)
        parsed[part] = blocks
        prefix = None
        sec = sub = 0
        fig_n = tab_n = eq_n = 0
        for b in blocks:
            if b["type"] == "heading":
                if b["level"] == 1:
                    sec = sub = fig_n = tab_n = eq_n = 0
                    if b["kind"] == "chapter":
                        chapter_no += 1
                        prefix = str(chapter_no)
                        b["number"] = prefix
                    elif b["kind"] == "appendix":
                        appendix_no += 1
                        prefix = chr(ord("A") + appendix_no - 1)
                        b["number"] = prefix
                    else:
                        prefix = None
                        b["number"] = None
                    if b.get("label"):
                        name = "Chapter" if b["kind"] == "chapter" else "Appendix"
                        labels[b["label"]] = f"{name} {prefix}"
                elif b["level"] == 2:
                    sec += 1
                    sub = 0
                    b["number"] = f"{prefix}.{sec}" if prefix and b["kind"] == "section" else None
                    if b.get("label"):
                        labels[b["label"]] = f"Section {b['number']}"
                elif b["level"] == 3:
                    sub += 1
                    b["number"] = f"{prefix}.{sec}.{sub}" if prefix else None
                    if b.get("label"):
                        labels[b["label"]] = f"Section {b['number']}"
                else:
                    b["number"] = None
            elif b["type"] == "fig":
                fig_n += 1
                num = f"{prefix}.{fig_n}"
                b["number"] = num
                labels[f"fig:{b['id']}"] = f"Figure {num}"
                expect = b["id"].split("_")[1] + "." + b["id"].split("_")[2]
                if expect != num:
                    problems.append(f"figure {b['id']} is numbered {num} (file suggests {expect})")
                if not (FIG / f"{b['id']}.png").exists():
                    problems.append(f"missing figure file {b['id']}")
                note_cites(b["caption_raw"])
                figures.append({"id": b["id"], "number": num, "caption_raw": b["caption_raw"]})
            elif b["type"] == "tab":
                tab_n += 1
                num = f"{prefix}.{tab_n}"
                b["number"] = num
                labels[f"tab:{b['id']}"] = f"Table {num}"
                expect = b["id"].split("_")[1] + "." + b["id"].split("_")[2]
                if expect != num:
                    problems.append(f"table {b['id']} is numbered {num} (file suggests {expect})")
                if b["id"] not in tables_spec:
                    problems.append(f"missing table spec {b['id']}")
                else:
                    for row in tables_spec[b["id"]]["rows"]:
                        for cell in row:
                            for key in re.findall(r"@([A-Za-z0-9_]+)", str(cell)):
                                if key in REFS and key not in cite_order:
                                    cite_order.append(key)
                tables.append({"id": b["id"], "number": num})
            elif b["type"] == "eq":
                eq_n += 1
                num = f"{prefix}.{eq_n}"
                b["number"] = num
                labels[f"eq:{b['id']}"] = f"Equation ({num})"
                if b["id"] not in EQUATIONS:
                    problems.append(f"missing equation {b['id']}")
                equations.append({"id": b["id"], "number": num})
            elif b["type"] in ("para", "quote"):
                note_cites(b["raw"])
            elif b["type"] in ("bullets", "numbered"):
                for it in b["raw_items"]:
                    note_cites(it)

    numbers = {key: i + 1 for i, key in enumerate(cite_order)}
    unused = [k for k in REFS if k not in numbers]

    def resolve(text: str) -> str:
        def cite(m):
            keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
            return ", ".join(f"[{numbers[k]}]" for k in keys if k in numbers)

        text = CITE_RE.sub(cite, text)

        def xref(m):
            key = f"{m.group(1)}:{m.group(2)}"
            if key not in labels:
                problems.append(f"unresolved reference {key}")
                return "??"
            return labels[key]

        return XREF_RE.sub(xref, text)

    # ---------- pass 2: resolve text into runs
    for part, title in PARTS:
        out_blocks = []
        for b in parsed[part]:
            t = b["type"]
            if t == "heading":
                out_blocks.append({"type": "heading", "level": b["level"], "kind": b["kind"],
                                   "number": b.get("number"), "text": b["title"]})
            elif t == "para":
                out_blocks.append({"type": "para", "runs": runs(resolve(b["raw"]))})
            elif t == "quote":
                out_blocks.append({"type": "quote", "runs": runs(resolve(b["raw"]))})
            elif t in ("bullets", "numbered"):
                out_blocks.append({"type": t, "items": [runs(resolve(x)) for x in b["raw_items"]]})
            elif t == "fig":
                out_blocks.append({"type": "figure", "id": b["id"], "path": str(FIG / f"{b['id']}.png"),
                                   "label": f"Figure {b['number']}", "caption_runs": runs(resolve(b["caption_raw"]))})
            elif t == "tab":
                spec = tables_spec[b["id"]]
                rows = [[pretty(make_tables.resolve_citations(str(c), numbers)) for c in row] for row in spec["rows"]]
                out_blocks.append({"type": "table", "id": b["id"], "label": f"Table {b['number']}",
                                   "caption": pretty(spec["caption"]), "columns": [pretty(c) for c in spec["columns"]], "rows": rows,
                                   "widths": spec["widths"], "note": pretty(spec.get("note")),
                                   "png": str(TAB / f"{b['id']}.png")})
            elif t == "eq":
                out_blocks.append({"type": "equation", "id": b["id"], "number": f"({b['number']})",
                                   "path": str(EQPNG / f"{b['id']}.png")})
            elif t == "code":
                out_blocks.append({"type": "code", "title": resolve(b["title"]), "lines": b["lines"]})
            elif t == "special":
                out_blocks.append({"type": "special", "name": b["name"]})
        model_parts.append({"part": part, "title": title, "blocks": out_blocks})

    # ---------- render equations and tables
    EQPNG.mkdir(exist_ok=True)
    plt.rcParams["mathtext.fontset"] = "cm"
    for key, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontsize=12)
        fig.savefig(EQPNG / f"{key}.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        plt.close(fig)
    for tid, spec in tables_spec.items():
        make_tables.render(tid, spec, numbers)
    (DATA / "tables.json").write_text(json.dumps(tables_spec, indent=2), encoding="utf-8")

    references = [{"n": numbers[k], "key": k, "text": ieee(k)} for k in cite_order]
    model = {
        "meta": {**META, "authors": [list(a) for a in META["authors"]], "supervisor": list(META["supervisor"]),
                 "coordinator": list(META["coordinator"]), "head": list(META["head"])},
        "nomenclature": NOMENCLATURE,
        "parts": model_parts,
        "figures": [{"id": f["id"], "number": f["number"], "caption": resolve(f["caption_raw"]),
                     "path": str(FIG / f"{f['id']}.png")} for f in figures],
        "tables": [{"id": t["id"], "number": t["number"], "caption": tables_spec[t["id"]]["caption"],
                    "path": str(TAB / f"{t['id']}.png")} for t in tables],
        "equations": equations,
        "references": references,
    }
    (DATA / "document_model.json").write_text(json.dumps(model, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---------- LaTeX support files
    LATEX.mkdir(exist_ok=True)
    (LATEX / "references.bib").write_text(bibtex(cite_order), encoding="utf-8")
    eq_lines = ["% Display equations of the P2 thesis. Labels match the Word numbering.", ""]
    for e in equations:
        eq_lines += [f"% Equation ({e['number']})", "\\begin{equation}", f"  {EQUATIONS[e['id']]}",
                     f"  \\label{{eq:{e['id']}}}", "\\end{equation}", ""]
    (LATEX / "equations.tex").write_text("\n".join(eq_lines), encoding="utf-8")
    ft = ["% Figure and table environments with captions and labels (graphicspath: ../figures/, ../tables/).", ""]
    for f in model["figures"]:
        cap = f["caption"].replace("%", "\\%").replace("&", "\\&")
        ft += [f"% Figure {f['number']}", "\\begin{figure}[htbp]", "  \\centering",
               f"  \\includegraphics[width=\\textwidth]{{{f['id']}.png}}", f"  \\caption{{{cap}}}",
               f"  \\label{{fig:{f['id']}}}", "\\end{figure}", ""]
    for t in model["tables"]:
        cap = t["caption"].replace("%", "\\%").replace("&", "\\&")
        ft += [f"% Table {t['number']} (rendered image; the editable rows are in data/tables.json and the Word files)",
               "\\begin{table}[htbp]", "  \\centering", f"  \\caption{{{cap}}}", f"  \\label{{tab:{t['id']}}}",
               f"  \\includegraphics[width=\\textwidth]{{{t['id']}.png}}", "\\end{table}", ""]
    (LATEX / "figures_and_tables.tex").write_text("\n".join(ft), encoding="utf-8")
    cite_map = ["% Citation number (Word/IEEE order) -> BibTeX key", ""] + [f"[{numbers[k]}]  {k}" for k in cite_order]
    (LATEX / "citation_key_map.txt").write_text("\n".join(cite_map), encoding="utf-8")

    words = 0
    for p in model_parts:
        for b in p["blocks"]:
            if b["type"] in ("para", "quote"):
                words += sum(len(r["text"].split()) for r in b["runs"])
            elif b["type"] in ("bullets", "numbered"):
                words += sum(len(r["text"].split()) for it in b["items"] for r in it)
    print(f"parts={len(model_parts)} figures={len(figures)} tables={len(tables)} equations={len(equations)} "
          f"references={len(references)} unused_refs={unused} words~{words}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)


if __name__ == "__main__":
    main()
