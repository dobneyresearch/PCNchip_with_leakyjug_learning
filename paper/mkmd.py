#!/usr/bin/env python3
"""Build an editable Markdown twin of main_stage2_v2.tex.

Figure captions are lifted from the .tex (not from pandoc's HTML) so their LaTeX
math survives intact. TikZ bodies cannot be represented in Markdown, so each
figure becomes a labelled placeholder carrying the editable caption.
"""
import re, subprocess, sys, pathlib

STEM = sys.argv[1] if len(sys.argv) > 1 else "main_stage2_v2"
TEX = pathlib.Path(f"{STEM}.tex")
OUT = pathlib.Path(f"{STEM}.md")

# ── 1. pandoc: markdown flavour (keeps @citekeys, {#ids}, $math$) + pipe tables ──
md = subprocess.run(
    ["pandoc", str(TEX), "-f", "latex",
     "-t", "markdown-simple_tables-multiline_tables-grid_tables+pipe_tables",
     "--standalone", "--wrap=none"],
    capture_output=True, text=True, check=True).stdout

# ── 2. figure captions straight from the .tex, with math preserved ──────────────
tex = TEX.read_text()
caps, order = {}, []
for fig in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.S):
    m_lab = re.search(r"\\label\{(fig:[^}]+)\}", fig)
    m_cap = re.search(r"\\caption\{(.*?)\}\s*\n\\label", fig, re.S)
    if not (m_lab and m_cap):
        continue
    cap = " ".join(m_cap.group(1).split())          # unwrap hard line breaks
    cap = cap.replace("\\textbf{", "**").replace("\\emph{", "*")
    # close the braces we just opened, in order
    out, depth = [], []
    i = 0
    while i < len(cap):
        if cap.startswith("**", i) and not depth:
            out.append("**"); depth.append("**"); i += 2; continue
        if cap[i] == "*" and not depth and not cap.startswith("**", i):
            out.append("*"); depth.append("*"); i += 1; continue
        if cap[i] == "}" and depth:
            out.append(depth.pop()); i += 1; continue
        out.append(cap[i]); i += 1
    caps[m_lab.group(1)] = "".join(out)
    order.append(m_lab.group(1))

NUM = {lab: n + 1 for n, lab in enumerate(order)}
WHAT = {
    "fig:chip": "Block diagram — the chip: W-SRAM at the centre, feeding (a) LUT -> weight DAC -> analog MAC -> ADC across the top, (b) the transpose engine reading the same SRAM, (c) the leaky jug writing +/-1 codes back into it.",
    "fig:net":  "Block diagram — the network: three L1 chips -> router (gather: sum then divide by fan-in) -> L2 chip, with the backward delta/partials shown dashed on the SAME links.",
    "fig:jug":  "Diagram — the leaky jug: a capacitor with +theta/-theta threshold lines, error flowing in at the top, a leak draining at the bottom right, and a fire arrow below.",
}

def fig_block(m):
    lab = m.group(1)
    n = NUM.get(lab, "?")
    return (f'<!-- FIGURE {n} ({lab}) — drawn in TikZ in {TEX.name}; not representable in Markdown.\n'
            f'     {WHAT.get(lab, "")}\n'
            f'     EDIT THE CAPTION BELOW FREELY. For changes to the DIAGRAM ITSELF, just say so in a note. -->\n\n'
            f'> **Figure {n} ({lab}).** {caps.get(lab, "")}\n')

md = re.sub(r'<figure id="(fig:[^"]+)">.*?</figure>', fig_block, md, flags=re.S)

# ── 3. resolve cross-references to real numbers, using the AUTHORITATIVE .aux ───
aux = pathlib.Path(f"{STEM}.aux")
labels = dict(re.findall(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}', aux.read_text())) if aux.exists() else {}
if not labels:
    sys.exit("ERROR: no .aux labels found — build the PDF first so refs resolve to numbers.")

UNRESOLVED = []

def deref(m):
    kind, lab = m.group(1), m.group(2)
    num = labels.get(lab)
    if num is None:                       # unknown label: keep it visible, never invent one
        UNRESOLVED.append(lab)
        return f"[{lab}?]"
    return f"({num})" if kind == "eqref" else num

# "Sec. [3](#sec:arch){reference-type="ref" reference="sec:arch"}" -> "Sec. 3"
# "[eq:bound](#eq:bound){reference-type="eqref" ...}"              -> "(7)"
# NB link text may itself contain escaped brackets ("[\[eq:jug\]]"), so match it lazily.
md = re.sub(r'\[.*?\]\(#[^)]+\)\{reference-type="(ref|eqref)"\s+reference="([^"]*)"\}',
            deref, md)

# ── 3b. expand the paper's private macros so the maths previews anywhere ────────
for frm, to in (("\\bdelta", "\\boldsymbol{\\delta}"), ("\\W", "\\boldsymbol{W}"),
                ("\\x", "\\boldsymbol{x}"), ("\\E", "\\boldsymbol{E}")):
    md = re.sub(re.escape(frm) + r'(?![A-Za-z])', lambda _m, t=to: t, md)

# ── 4. unwrap the ::: table divs, keeping the caption line ──────────────────────
md = re.sub(r'::: \{#(tab:[^}]+)\}\n', lambda m: f'<!-- TABLE {m.group(1)} -->\n', md)
md = re.sub(r'\n:::\n', '\n', md)

# ── 5. editing header ──────────────────────────────────────────────────────────
header = f"""<!-- ============================================================================
EDITABLE MARKDOWN TWIN of {TEX.name}  (generated {subprocess.run(['date','+%Y-%m-%d'],capture_output=True,text=True).stdout.strip()})

The .tex remains the SOURCE OF TRUTH for the PDF. Edit this file freely and hand
it back; I will port your edits into the .tex and rebuild the PDF.

  * Prose, headings, tables, captions  -> edit directly here.
  * [@citekey]                         -> citations; keys live in refs.bib.
  * {{#sec:foo}} / <!-- TABLE tab:foo -->  -> anchors I use to map edits back. Leave them if you can,
                                          but don't worry if they get mangled.
  * $...$ / $$...$$                    -> LaTeX maths, passed through verbatim.
  * FIGURES are TikZ drawings and cannot round-trip through Markdown. Their
    CAPTIONS are here and editable; for changes to a diagram itself, leave a note.

To regenerate this file from the .tex:  python3 mkmd.py
============================================================================ -->

"""
OUT.write_text(header + md)
print(f"wrote {OUT}  ({len(md.splitlines())} lines)")
for lab in order:
    print(f"  figure {NUM[lab]:>2}  {lab:<10} caption {len(caps.get(lab,''))} chars")

leftover = re.findall(r'reference-type=|<figure|<figcaption|^::: ', md, re.M)
print(f"  unresolved refs : {sorted(set(UNRESOLVED)) or 'none'}")
print(f"  leftover markup : {len(leftover)}")
if leftover or UNRESOLVED:
    sys.exit("WARNING: conversion left artefacts — inspect before sending.")
