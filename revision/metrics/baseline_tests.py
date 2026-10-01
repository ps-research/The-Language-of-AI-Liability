"""Baseline and second-pair comparisons, with the notebooks' own test code.

Pairs: each main-corpus AI law vs its jurisdiction's non-AI baseline, plus
Colorado SB 24-205 vs its successor SB 26-189 (parallel to SB 1047 vs SB 53).
Dimensions: modal composition (chi-square, Cramer's V), voice (Fisher, active vs
passive), actor categories (chi-square or Fisher, v2 classifier), tree depth,
conditional density, hedge density (permutation test, Cohen's d).
"""
import json
import re
from pathlib import Path

import numpy as np
import spacy
from scipy.stats import chi2_contingency, fisher_exact

REPO = Path(__file__).resolve().parents[2]  # repository root
CODE, OUT = REPO, REPO / "revision" / "metrics"
cells = lambda nb: ["".join(c["source"]) for c in json.load(open(CODE / nb))["cells"]]
DM, OC, RS = cells("Deontic_Modal_Analysis.ipynb"), cells("Obligation_Complexity_Analysis.ipynb"), cells("Regulatory_Specificity_Index.ipynb")

ns = {"np": np, "re": re, "chi2_contingency": chi2_contingency}
src = DM[7]; exec(src[src.index("def cramers_v"):src.index("\n\n", src.index("def cramers_v"))], ns)
src = OC[11]
for name in ("def permutation_test", "def cohens_d"):
    i = src.index(name); exec(src[i:src.index("\n\n\n", i) if "\n\n\n" in src[i:] else None], ns)
exec(RS[6].split('print("\\nComputing')[0], ns)    # HEDGE_WORDS
exec(OC[5].split('print("\\nComputing')[0], ns)    # CONDITIONAL_MARKERS

res = {**json.load(open(CODE / "Data/processed/preprocessed_corpus.json")),
       **json.load(open(REPO / "revision/preprocess/extra_corpus.json"))}
ext = json.load(open(OUT / "metrics_extended.json"))
v2 = json.load(open(REPO / "revision/t2/technique2_responsibility_v2.json"))
v1_voice = json.load(open(REPO / "revision/t2/v1_voice.json"))

PAIRS = [
    ("EU: AI Act vs DSA", "eu_ai_act", "eu_dsa"),
    ("China: GenAI Measures vs PIPL", "china_genai", "cn_pipl"),
    ("S. Korea: AI Basic Act vs PIPA", "south_korea", "kr_pipa"),
    ("Colorado: SB 24-205 vs Privacy Act", "co_sb205", "co_sb21_190"),
    ("Colorado: SB 24-205 vs SB 26-189", "co_sb205", "co_sb26_189"),
]
LAWS = sorted({l for _, a, b in PAIRS for l in (a, b)})

# per-sentence tree depths: notebook T3 cell 4, verbatim, on these laws
spacy.prefer_gpu()
dns = {"results": res, "LAW_ORDER": LAWS, "np": np}
setup = OC[3].replace('Path("/content/drive/MyDrive/ACL SRW TEXTS")', f'Path("{CODE / "Data"}")')
exec(setup, dns); dns["results"], dns["LAW_ORDER"] = res, LAWS
exec(OC[4], dns)
depths = {l: np.array(dns["depth_results"][l]["depths"]) for l in LAWS}

CATS4 = ["HARD_OBLIGATION", "SOFT_OBLIGATION", "PERMISSION", "PROHIBITION"]
REAL = ["regulated_entity", "government", "generic"]


def voice(l):
    if l in ext["t2_new"]:
        return ext["t2_new"][l]["voice"]
    c = [r["voice"] for r in v1_voice if r["law_id"] == l]
    return {k: c.count(k) for k in ("active", "passive", "indeterminate")}


def actors(l):
    cc = ext["t2_new"][l]["category_counts"] if l in ext["t2_new"] else v2[l]["category_counts"]
    return [cc.get(c, 0) for c in REAL]


def per_sent(l, pat):
    return np.array([len(pat.findall(s["text"])) for s in res[l]["sentences"] if s["is_obligation"]])


def cat_test(ra, rb):
    t = np.array([ra, rb]); t = t[:, t.sum(axis=0) > 0]
    if t.shape[1] < 2:
        return {"test": "n/a"}
    if t.shape[1] == 2:
        orr, p = fisher_exact(t); return {"test": "Fisher", "stat": float(orr), "p": float(p)}
    chi2, p, dof, exp = chi2_contingency(t)
    return {"test": "chi2", "stat": float(chi2), "p": float(p), "V": float(ns["cramers_v"](t)), "min_exp": float(exp.min())}


def num_test(a, b):
    diff, p = ns["permutation_test"](a, b)
    return {"a": float(a.mean()), "b": float(b.mean()), "p": float(p), "d": float(ns["cohens_d"](a, b))}


out = []
for name, a, b in PAIRS:
    va, vb = voice(a), voice(b)
    r = {"pair": name, "a": a, "b": b,
         "modal": cat_test([res[a]["stats"]["deontic_counts"].get(c, 0) for c in CATS4],
                           [res[b]["stats"]["deontic_counts"].get(c, 0) for c in CATS4]),
         "voice": dict(zip(("OR", "p"), map(float, fisher_exact([[va["active"], va["passive"]], [vb["active"], vb["passive"]]])))),
         "active_pct": [va["active"] / sum(va.values()), vb["active"] / sum(vb.values())],
         "actors": cat_test(actors(a), actors(b)), "actor_counts": [actors(a), actors(b)],
         "depth": num_test(depths[a], depths[b]),
         "conditional": num_test(per_sent(a, ns["CONDITIONAL_MARKERS"]), per_sent(b, ns["CONDITIONAL_MARKERS"])),
         "hedge": num_test(per_sent(a, ns["HEDGE_WORDS"]), per_sent(b, ns["HEDGE_WORDS"]))}
    out.append(r)
json.dump(out, open(OUT / "baseline_tests.json", "w"), indent=1)

f = lambda t: "n/a" if t.get("test") == "n/a" else f"p={t['p']:.3f}"
print(f"{'pair':<36} {'modal':>9} {'voice':>9} {'actors':>9} | {'depth a/b p':>17} | {'cond a/b p':>17} | {'hedge a/b p':>17}")
for r in out:
    print(f"{r['pair']:<36} {f(r['modal']):>9} {'p=%.3f' % r['voice']['p']:>9} {f(r['actors']):>9} | "
          + " | ".join(f"{r[k]['a']:.2f}/{r[k]['b']:.2f} p={r[k]['p']:.3f}" for k in ("depth", "conditional", "hedge")))
