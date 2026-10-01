"""Appendix D frequency tables, counted exactly as the metrics count them.

Uses the compiled patterns from the metric notebooks themselves
(Regulatory_Specificity_Index: HEDGE_WORDS, PRECISE_TEMPORAL, VAGUE_TEMPORAL;
Obligation_Complexity_Analysis: CONDITIONAL_MARKERS), applied with findall to
the obligation sentences, so each span is counted once. Word_List.ipynb counted
each pattern separately, so multiword items ("as appropriate", "provided that")
were also counted under their single-word part.
"""
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # repository root
OUT = REPO / "revision" / "t3"


def grab(nb, name):
    """Exec the `NAME = re.compile(...)` statement from a notebook, verbatim."""
    for c in json.load(open(REPO / nb))["cells"]:
        s = "".join(c["source"])
        i = s.find(f"{name} = re.compile(")
        if i >= 0:
            j = s.index("\n)", i) + 2
            ns = {"re": re}
            exec(s[i:j], ns)
            return ns[name]
    raise KeyError(name)


PAT = {
    "hedge": grab("Regulatory_Specificity_Index.ipynb", "HEDGE_WORDS"),
    "precise": grab("Regulatory_Specificity_Index.ipynb", "PRECISE_TEMPORAL"),
    "vague": grab("Regulatory_Specificity_Index.ipynb", "VAGUE_TEMPORAL"),
    "conditional": grab("Obligation_Complexity_Analysis.ipynb", "CONDITIONAL_MARKERS"),
}
MON = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"


def label(kind, m):
    m = re.sub(r"\s+", " ", m.strip().lower())
    if kind == "precise":
        if re.fullmatch(MON + r" \d+,? \d{4}", m): return "full date (Month D, YYYY)"
        if m.startswith("on or before"): return "on or before [Month D]"
        if m.startswith("by "): return "by [Month D] / by [date]"
        if m.startswith("effective"): return "effective [Month D]"
        if m.startswith("within"): return "within N days/hours/weeks/months/years"
        if m.startswith("every"): return "every N days/months/years"
    if kind == "vague" and m.startswith("without") and "delay" in m: return "without (undue) delay"
    if kind == "conditional" and m.startswith("except"): return "except when/where/that/as"
    if kind == "conditional" and m.startswith("in ") and "case" in m: return "in (the) case of/that/where"
    return m


results = json.load(open(REPO / "Data" / "processed" / "preprocessed_corpus.json"))
obl = [s["text"] for law in results.values() for s in law["sentences"] if s["is_obligation"]]
tables = {}
for kind, pat in PAT.items():
    c = Counter(label(kind, m) for t in obl for m in pat.findall(t))
    tables[kind] = dict(c.most_common())
    print(f"\n== {kind}: total {sum(c.values())} over {len(obl)} obligation sentences ==")
    for k, v in c.most_common():
        print(f"  {k:<40}{v:>5}  {100*v/sum(c.values()):5.1f}%")
json.dump(tables, open(OUT / "appendixD_tables_v2.json", "w"), indent=2)
