#!/usr/bin/env python3
"""Assemble main_stage2_v4.tex from v4.md (prose) + v3.tex (math/figures/preamble).
Equations, the 3 TikZ figures, and the preamble/macros are taken verbatim from
v3.tex so numbered-equation \\eqref-style references stay correct; all prose comes
from the author-approved v4.md via pandoc."""
import re, subprocess, pathlib

HERE = pathlib.Path(__file__).parent
tex3 = (HERE / "main_stage2_v3.tex").read_text()
md4 = (HERE / "main_stage2_v4.md").read_text()

# ---- 1. preamble (everything up to and including \maketitle), title fixed ----
pre = tex3.split(r"\begin{document}")[0]
pre = pre.replace("Analog Substrate for", "Analog Architecture for")  # md title

# ---- 2. pandoc table-support macros (generate + lift the preamble block) ----
sample = "| a | b |\n|---|---|\n| 1 | 2 |\n\n: cap\n"
std = subprocess.run(["pandoc", "-f", "markdown", "-t", "latex", "-s"],
                     input=sample, capture_output=True, text=True).stdout
tbl = "\n".join(l for l in std.splitlines()
                if any(k in l for k in ("longtable", "array}", "booktabs",
                                        "calc}", "\\real", "minipage",
                                        "\\newcommand{\\real")))
# a robust minimal set:
tbl_pre = (r"\usepackage{longtable,booktabs,array}" "\n"
           r"\usepackage{calc}" "\n"
           r"\providecommand{\tightlist}{"
           r"\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}" "\n"
           r"\providecommand{\passthrough}[1]{#1}" "\n"
           r"\providecommand{\pandocbounded}[1]{#1}" "\n")

# ---- 3. extract the numbered equation environments (in order) from v3.tex ----
eqs = [m.group(0) for m in
       re.finditer(r"\\begin\{(equation|align)\}.*?\\end\{\1\}", tex3, re.S)]
# ---- extract the 3 figure environments, keyed by label ----
figs = {}
for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", tex3, re.S):
    lab = re.search(r"\\label\{(fig:[^}]+)\}", m.group(0))
    if lab:
        figs[lab.group(1)] = m.group(0)

# ---- 4. abstract + body via pandoc ----
def pandoc(text):
    r = subprocess.run(["pandoc", "-f", "markdown", "-t", "latex", "--natbib"],
                       input=text, capture_output=True, text=True)
    out = r.stdout
    out = out.replace(r"\citep{", r"\cite{").replace(r"\citet{", r"\cite{")
    out = out.replace(r"\autocite{", r"\cite{")
    return out

# abstract text sits in the YAML block  abstract: | ... (indented)
ab = re.search(r"\nabstract:\s*\|\n(.*?)\nauthor:", md4, re.S).group(1)
ab = "\n".join(l[2:] if l.startswith("  ") else l for l in ab.splitlines())
abstract_tex = pandoc(ab).strip()

# body = everything after the YAML front-matter (second '---')
body_md = md4.split("\n---\n", 1)[1]
body_md = body_md.split("---", 1)[1] if body_md.lstrip().startswith("-") else body_md
# strip the leading YAML that remains up to first '# '
body_md = body_md[body_md.index("\n# "):]
body = pandoc(body_md)

# ---- 5. splice equations: replace pandoc \[...\] blocks in order ----
disp = list(re.finditer(r"\\\[.*?\\\]", body, re.S))
assert len(disp) == len(eqs), f"{len(disp)} display vs {len(eqs)} eqs"
for m, eq in zip(reversed(disp), reversed(eqs)):
    body = body[:m.start()] + eq + body[m.end():]

# ---- 6. splice figures: replace each Figure-caption quote block ----
def fig0():
    tikz = r"""\begin{figure}[t]
\centering
\begin{tikzpicture}[node distance=7mm and 12mm,font=\footnotesize]
  \node[ana] (w1) {$\bm W_1$};
  \node[ana,right=of w1] (w2) {$\bm W_2$};
  \node[dig,right=of w2] (ro) {read-out\\(forms $\bm\delta$)};
  \node[left=9mm of w1,font=\scriptsize] (in) {input};
  \draw[flow] (in) -- (w1);
  \draw[flow] (w1) -- (w2);
  \draw[flow] (w2) -- (ro);
  \node[jug,below=10mm of w1] (e1) {$\bm E_1$ (jug)};
  \node[jug,below=10mm of w2] (e2) {$\bm E_2$ (jug)};
  \draw[flow] (ro.south) |- (e2.east) node[near end,above,font=\scriptsize] {$\bm W_2^{\!\top}\bm\delta$};
  \draw[flow] (e2.west) -- (e1.east) node[midway,above,font=\scriptsize] {$\bm W_1^{\!\top}\bm\delta$};
  \draw[flow,red!60!black] (e1.north) -- (w1.south) node[midway,right,font=\scriptsize] {$\pm1$};
  \draw[flow,red!60!black] (e2.north) -- (w2.south) node[midway,right,font=\scriptsize] {$\pm1$};
  \begin{scope}[on background layer]
    \node[draw,dashed,rounded corners,fit=(w1)(ro)(e1)(e2)(in),inner sep=7pt,
          label={[font=\scriptsize\itshape,text=gray]below:all links via the router}] {};
  \end{scope}
\end{tikzpicture}
\caption{\textbf{The learning loop in one picture.} Forward activations pass
input $\to \bm W_1 \to \bm W_2 \to$ read-out, which forms the output error
$\bm\delta$. The error is relayed \emph{backward through the network but forward
in signalling}: each layer computes $\bm W^{\!\top}\bm\delta$ on the chip that
owns $\bm W$ and injects it into that layer's error store $\bm E$ (a leaky jug),
which writes single-code updates into $\bm W$. All links are mediated by the
router. No analog signal flows in reverse.}
\label{fig:overview}
\end{figure}"""
    return tikz

# pandoc renders the blockquote captions as \begin{quote}...\end{quote}
def replace_fig(body, label, repl):
    pat = re.compile(r"\\begin\{quote\}(?:(?!\\end\{quote\}).)*?" +
                     re.escape("(" + label + ")") + r".*?\\end\{quote\}", re.S)
    return pat.sub(lambda _: repl, body, count=1)

body = replace_fig(body, "fig:overview", fig0())
for lab in ("fig:chip", "fig:net", "fig:jug"):
    body = replace_fig(body, lab, figs[lab])

# adding the overview as the first figure shifts numbering; make the literal
# prose references track the labels (overview=1, chip=2, net=3, jug=4).
body = body.replace("Fig. 1", r"Fig.~\ref{fig:chip}")
body = body.replace("Fig. 2", r"Fig.~\ref{fig:net}")
body = body.replace("Fig. 3", r"Fig.~\ref{fig:jug}")

# convert the (now-correct) literal equation references to \eqref so numbering
# cannot silently rot on future edits.  Every bare "(N)" in the prose is an
# equation reference (verified); section refs are left as literals for now.
for lit, lab in (("(4)", "eq:drec"), ("(5)", "eq:jug"),
                 ("(6)", "eq:invariant"), ("(7)", "eq:bound")):
    body = body.replace(lit, r"\eqref{%s}" % lab)

# tidy: drop any stray raw HTML comments pandoc passed through
body = re.sub(r"\\begin\{verbatim\}<!--.*?-->\\end\{verbatim\}", "", body, flags=re.S)

# ---- 7. assemble ----
doc = (pre
       + tbl_pre
       + "\\begin{document}\n\\maketitle\n\n"
       + "\\begin{abstract}\n" + abstract_tex + "\n\\end{abstract}\n\n"
       + body
       + "\n\n\\bibliographystyle{IEEEtran}\n\\bibliography{refs}\n"
       + "\\end{document}\n")
# normalize exotic unicode spaces pandoc/mkmd left in the prose
for cp in (0x2006,0x2009,0x2005,0x2007,0x2008,0x202F,0x2002,0x2003,0x00A0,0x2004,0x200A,0x2028):
    doc = doc.replace(chr(cp), " ")
doc = doc.replace(chr(0x2011), "-")
(HERE / "main_stage2_v4.tex").write_text(doc)
print("wrote main_stage2_v4.tex  |  eqs spliced:", len(eqs),
      "| figs:", list(figs) + ["fig:overview"])
