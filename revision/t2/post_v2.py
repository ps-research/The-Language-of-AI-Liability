"""After t2_v2.py: re-run the notebook's own statistics cell (cell 12) and the
notebook's own Figure 3b code (from cell 10) on the v2 actor results.

Voice results are the reproduced v1 ones (voice classification is unchanged),
so the voice tests in cell 12 should reproduce Appendix E exactly; only the
actor-category tests change. HHI in the figure is the category-level HHI.
"""
import contextlib
import io
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]  # repository root
OUT = REPO / "revision" / "t2"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

cells = json.load(open(REPO / "Responsibility_Attribution_Analysis.ipynb"))["cells"]
setup = "".join(cells[3]["source"])
stats_src = "".join(cells[12]["source"])
fig_src = "".join(cells[10]["source"])
fig_src = fig_src[fig_src.index("# ── Figure 3b"):]
fig_src = fig_src.replace('FIGURE_DIR / "fig3b_actor_concentration.pdf"',
                          'FIGURE_DIR / "fig2b_actor_concentration.pdf"')

# LAW_ORDER / LAW_LABELS exactly as the notebook defines them
ns = {}
lab_src = setup[setup.index("LAW_ORDER = ["):setup.index('print(f"Loaded')]
exec(lab_src, ns)
LAW_ORDER, LAW_LABELS = ns["LAW_ORDER"], ns["LAW_LABELS"]

v2 = json.load(open(OUT / "technique2_responsibility_v2.json"))
voice_rows = json.load(open(OUT / "v1_voice.json"))

voice_results = {}
for law_id in LAW_ORDER:
    c = Counter(r["voice"] for r in voice_rows if r["law_id"] == law_id)
    voice_results[law_id] = {"counts": {k: c.get(k, 0) for k in ("active", "passive", "indeterminate")},
                             "total_obligation": sum(c.values())}

actor_results = {}
for law_id in LAW_ORDER:
    p = v2[law_id]
    cc = dict(p["category_counts"])
    actor_results[law_id] = {
        "category_counts": cc,
        "total_active": p["total_active"],
        "hhi": p["hhi_category"],
        "classified_real_actors": p["classified_real_actors"],
    }

# Figure data: share of active-voice sentences per category (as cell 10 does)
DISPLAY_CATS = ["regulated_entity", "government", "generic", "anaphoric", "non_actor"]
cat_df = pd.DataFrame({
    lid: {c: actor_results[lid]["category_counts"].get(c, 0) / actor_results[lid]["total_active"] * 100
          for c in DISPLAY_CATS}
    for lid in LAW_ORDER}).T.loc[LAW_ORDER]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
})

common = dict(np=np, plt=plt, mticker=mticker, pd=pd, LAW_ORDER=LAW_ORDER, LAW_LABELS=LAW_LABELS,
              voice_results=voice_results, actor_results=actor_results, cat_df=cat_df,
              FIGURE_DIR=FIG)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(stats_src, "nb_cell_12", "exec"), dict(common))
(OUT / "stats_v2.txt").write_text(buf.getvalue())
print(buf.getvalue())

exec(compile(fig_src, "nb_cell_10_fig3b", "exec"), dict(common))

print("\nPer-law category shares (% of active-voice obligation sentences):")
print(cat_df[["regulated_entity", "government", "generic", "anaphoric"]].round(1).to_string())
print("\nShares among classified real actors + HHI:")
for lid in LAW_ORDER:
    p = v2[lid]
    sh = {k: round(v * 100, 1) for k, v in p["category_shares"].items()}
    print(f"  {LAW_LABELS[lid]:<26} n={p['classified_real_actors']:>3} HHI_cat={p['hhi_category']:.2f} "
          f"(v1 label-HHI was in paper) shares={sh}")
