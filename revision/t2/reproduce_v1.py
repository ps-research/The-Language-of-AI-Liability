"""Re-run Technique 2 cells (3-6, 9) of Responsibility_Attribution_Analysis.ipynb verbatim.

Only change: the Colab BASE_DIR is pointed at the local clone's Data/ folder.
Writes per-sentence v1 extractions (needed for the v1-vs-v2 diff) and checks
the per-law outputs against the saved technique2_responsibility.json.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # repository root
NB = REPO / "Responsibility_Attribution_Analysis.ipynb"
OUT = REPO / "revision" / "t2"

cells = json.load(open(NB))["cells"]
src = {i: "".join(cells[i]["source"]) for i in (3, 4, 5, 6, 9)}
src[3] = src[3].replace('Path("/content/drive/MyDrive/ACL SRW TEXTS")',
                        f'Path("{REPO / "Data"}")')
assert str(REPO / "Data") in src[3], "BASE_DIR patch failed"

ns = {}
for i in (3, 4, 5, 6, 9):
    print(f"--- running notebook cell {i} ---", flush=True)
    exec(compile(src[i], f"nb_cell_{i}", "exec"), ns)

saved = json.load(open(REPO / "Data" / "processed" / "technique2_responsibility.json"))
ok = True
for law_id in ns["LAW_ORDER"]:
    ar, vr = ns["actor_results"][law_id], ns["voice_results"][law_id]
    got = {
        "voice": vr["counts"],
        "category_counts": dict(ar["category_counts"]),
        "hhi": round(ar["hhi"], 6),
        "classified_real_actors": ar.get("classified_real_actors"),
    }
    want = {
        "voice": saved[law_id]["voice"],
        "category_counts": saved[law_id]["category_counts"],
        "hhi": round(saved[law_id]["hhi"], 6),
        "classified_real_actors": saved[law_id]["classified_real_actors"],
    }
    match = got == want
    ok &= match
    print(f"{law_id:<12} {'MATCH' if match else 'DIFF'}" + ("" if match else f"\n   got  {got}\n   want {want}"))

rows = []
for law_id in ns["LAW_ORDER"]:
    for e in ns["actor_results"][law_id]["extractions"]:
        rows.append({"law_id": law_id, "sent_id": e["sent_id"], "subject": e["subject"],
                     "v1_label": e["actor_label"], "v1_category": e["actor_category"]})
voice_rows = [{"law_id": l, "sent_id": s["sent_id"], "voice": s["voice"], "text": s["text"]}
              for l in ns["LAW_ORDER"] for s in ns["voice_results"][l]["sentences"]]
json.dump(rows, open(OUT / "v1_extractions.json", "w"), indent=1)
json.dump(voice_rows, open(OUT / "v1_voice.json", "w"), indent=1)
print("ALL MATCH" if ok else "SOME DIFF", "| wrote", len(rows), "extractions,", len(voice_rows), "voice rows")
sys.exit(0 if ok else 1)
