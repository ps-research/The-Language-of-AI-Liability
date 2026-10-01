"""Techniques 2-4 on the extended corpus (10 original + 4 AI + 4 non-AI baselines).

T3 and T4: the notebooks' own computation cells are executed verbatim with
LAW_ORDER / LAW_LABELS / results extended; the 10 original laws are checked
against the saved technique3_complexity.json / technique4_specificity.json.
T2 for the 8 new laws: voice with the notebook's classify_voice (cell 4,
verbatim) and actors with the approved v2 classifier (work/t2/t2_v2.py).
"""
import importlib.util
import json
import re
from pathlib import Path

import spacy

REPO = Path(__file__).resolve().parents[2]  # repository root
CODE, OUT = REPO, REPO / "revision" / "metrics"
spacy.prefer_gpu()

orig = json.load(open(CODE / "Data" / "processed" / "preprocessed_corpus.json"))
extra = json.load(open(REPO / "revision" / "preprocess" / "extra_corpus.json"))
results = {**orig, **extra}
ORIG = list(orig)
NEW_AI = ["cn_algorithms", "cn_deep_synthesis", "cn_ai_labeling", "co_sb26_189"]
BASE = ["eu_dsa", "cn_pipl", "kr_pipa", "co_sb21_190"]
ALL = ORIG + NEW_AI + BASE
LABELS_NEW = {
    "cn_algorithms": "China Algorithmic Rec.", "cn_deep_synthesis": "China Deep Synthesis",
    "cn_ai_labeling": "China AI Labeling", "co_sb26_189": "CO SB 26-189",
    "eu_dsa": "EU DSA", "cn_pipl": "China PIPL", "kr_pipa": "South Korea PIPA",
    "co_sb21_190": "CO Privacy Act",
}


def nb_cells(name):
    return ["".join(c["source"]) for c in json.load(open(CODE / name))["cells"]]


def run_notebook(name, setup_idx, compute_idx):
    cells = nb_cells(name)
    setup = cells[setup_idx].replace('Path("/content/drive/MyDrive/ACL SRW TEXTS")', f'Path("{CODE / "Data"}")')
    ns = {}
    exec(setup, ns)
    ns["LAW_LABELS"] = {**ns["LAW_LABELS"], **LABELS_NEW}
    ns["LAW_ORDER"] = ALL
    ns["results"] = results
    if "CORPUS_FILES" in ns:   # raw text files (cross-reference density reads them)
        for lid in NEW_AI + BASE:
            ns["CORPUS_FILES"][lid] = str(REPO / "revision" / "corpus_ext" / "texts" / f"{lid}.txt")
    if "nlp" in ns:
        ns["nlp"].max_length = 700_000
    for i in compute_idx:
        exec(cells[i], ns)
    return ns


def close(a, b):
    return abs(float(a) - float(b)) < 1e-9


# ── T3 ──
t3 = run_notebook("Obligation_Complexity_Analysis.ipynb", 3, [4, 5, 6, 7])
saved3 = json.load(open(CODE / "Data" / "processed" / "technique3_complexity.json"))
for lid in ORIG:
    s = saved3[lid]
    ok = (close(s["depth"]["mean"], t3["depth_results"][lid]["mean"]) if "mean" in s["depth"] else True) \
        and s["conditional"]["total"] == t3["conditional_results"][lid]["total_conditionals"] \
        and close(s["crossref"]["density"], t3["crossref_results"][lid]["density"])
    print(f"T3 {lid:<12} {'MATCH' if ok else 'DIFF'}")

# ── T4 ──
t4 = run_notebook("Regulatory_Specificity_Index.ipynb", 2, [3, 4, 5, 6, 7])
saved4 = json.load(open(CODE / "Data" / "processed" / "technique4_specificity.json"))
for lid in ORIG:
    s = saved4[lid]
    ok = close(s["numeric_precision"]["ratio"], t4["numeric_results"][lid]["ratio"]) \
        and close(s["definitional_density"]["density"], t4["definitional_results"][lid]["density"]) \
        and close(s["temporal_precision"]["ratio"], t4["temporal_results"][lid]["ratio"]) \
        and close(s["hedge_density"]["density"], t4["hedge_results"][lid]["density"])
    print(f"T4 {lid:<12} {'MATCH' if ok else 'DIFF'}")

# ── T2 for the 8 new laws ──
spec = importlib.util.spec_from_file_location("t2v2", REPO / "revision" / "t2" / "t2_v2.py")
t2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t2)
ra = nb_cells("Responsibility_Attribution_Analysis.ipynb")
voice_src = ra[4]
voice_ns = {"results": results, "LAW_ORDER": NEW_AI + BASE, "nlp": t3["nlp"]}
exec(voice_src, voice_ns)
real = ("regulated_entity", "government", "generic")
t2_out = {}
for lid in NEW_AI + BASE:
    vr = voice_ns["voice_results"][lid]
    active = [s for s in vr["sentences"] if s["voice"] == "active"]
    cats = {c: 0 for c in (*real, "anaphoric", "unclassified")}
    rows = []
    for s, doc in zip(active, t3["nlp"].pipe([s["text"] for s in active], batch_size=16)):
        _, words = t2.head_phrase(doc)
        label, cat = t2.classify_words(words)
        cats[cat] += 1
        rows.append({"sent_id": s["sent_id"], "head_phrase": " ".join(words), "label": label, "category": cat})
    n = sum(cats[c] for c in real)
    shares = {c: cats[c] / n for c in real} if n else {}
    t2_out[lid] = {"voice": vr["counts"], "total_obligation": vr["total_obligation"],
                   "total_active": len(active), "category_counts": cats,
                   "classified_real_actors": n, "category_shares": shares,
                   "hhi_category": sum(v * v for v in shares.values()), "rows": rows}

# ── Save ──
def pick(d, keys):
    return {k: d[k] for k in keys if k in d}

json.dump({
    "law_order": ALL, "labels": {**t3["LAW_LABELS"]}, "groups": {"orig": ORIG, "new_ai": NEW_AI, "baseline": BASE},
    "t3": {lid: {"depth": pick(t3["depth_results"][lid], ["mean", "median", "n"]),
                 "conditional": pick(t3["conditional_results"][lid], ["total_conditionals", "n_obligation", "density"]),
                 "crossref": pick(t3["crossref_results"][lid], ["total_refs", "total_sentences", "density"])} for lid in ALL},
    "t4": {lid: {"numeric": pick(t4["numeric_results"][lid], ["ratio", "sents_with_numeric", "n_obligation"]),
                 "definitional": pick(t4["definitional_results"][lid], ["density", "defined_terms", "recurring_nps"]),
                 "temporal": pick(t4["temporal_results"][lid], ["ratio", "precise", "vague", "total"]),
                 "hedge": pick(t4["hedge_results"][lid], ["density", "total_hedges", "n_obligation"])} for lid in ALL},
    "t2_new": t2_out,
}, open(OUT / "metrics_extended.json", "w"), indent=1)

print("\nlaw                    oblig  depth  cond  xref | num   def   temp  hedge | act%  HHI_cat  n_real")
for lid in NEW_AI + BASE:
    a, b, c = t3, t4, t2_out[lid]
    v = c["voice"]; tot = sum(v.values())
    print(f"{t3['LAW_LABELS'][lid]:<22} {results[lid]['stats']['obligation_sentences']:>5} "
          f"{a['depth_results'][lid]['mean']:>6.2f} {a['conditional_results'][lid]['density']:>5.2f} "
          f"{a['crossref_results'][lid]['density']:>5.2f} | {b['numeric_results'][lid]['ratio']:>4.2f} "
          f"{b['definitional_results'][lid]['density']:>5.2f} {b['temporal_results'][lid]['ratio']:>5.2f} "
          f"{b['hedge_results'][lid]['density']:>5.2f} | {100*v['active']/tot:>4.0f} {c['hhi_category']:>7.2f} "
          f"{c['classified_real_actors']:>6}")
