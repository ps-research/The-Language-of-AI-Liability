"""Preprocessing, exactly as in "Notebook 1 : Data Preprocessing" (final cell 14).

The definitions (definitional patterns, deontic Matcher, resolve_matches,
process_law with min_tokens=4, CORPUS, METADATA) are executed verbatim from the
notebook copy in this folder. Only the input/output locations differ.

    python preprocess.py verify          # re-run the 10 original laws, compare
                                         # with code/Data/processed/preprocessed_corpus.json
    python preprocess.py run MANIFEST    # process extra laws listed in a JSON
                                         # manifest {law_id: {"file":..., "metadata":{...}}}
"""
import json
import sys
from pathlib import Path

import spacy
from spacy.matcher import Matcher  # noqa: F401  (used by the exec'd notebook code)

REPO = Path(__file__).resolve().parents[2]  # repository root
HERE = REPO / "revision" / "preprocess"
DATA = REPO / "Data"

cells = json.load(open(HERE / "Notebook_1_Data_Preprocessing.ipynb"))["cells"]
src = "".join(cells[14]["source"])
start = src.index("DEFINITION_HEADER_PATTERNS = re.compile(")
end = src.index("matcher = build_deontic_matcher(nlp.vocab)")
ns = {}
exec("import re\nfrom spacy.matcher import Matcher\nfrom collections import defaultdict\n"
     + src[start:end], ns)
MIN_TOKENS = 4  # set in the notebook's Fix 2 block


def load_nlp():
    if spacy.prefer_gpu():
        print("spaCy on GPU", flush=True)
    nlp = spacy.load("en_core_web_trf")
    nlp.max_length = 700_000
    return nlp


def verify():
    nlp = load_nlp()
    matcher = ns["build_deontic_matcher"](nlp.vocab)
    saved = json.load(open(DATA / "processed" / "preprocessed_corpus.json"))
    out, all_ok = {}, True
    for law_id, filename in ns["CORPUS"].items():
        text = (DATA / filename).read_text(encoding="utf-8")
        r = ns["process_law"](law_id, text, nlp, matcher, min_tokens=MIN_TOKENS)
        out[law_id] = r
        s, o = saved[law_id], r
        key = lambda x: (x["sent_id"], x["text"], x["is_obligation"], x["is_definitional"],
                         sorted(x["deontic_categories"]),
                         [(m["category"], m["span"]) for m in x["deontic_matches"]])
        a, b = [key(x) for x in s["sentences"]], [key(x) for x in o["sentences"]]
        same_sents = a == b
        same_stats = s["stats"] == o["stats"]
        ok = same_sents and same_stats
        all_ok &= ok
        n_diff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        print(f"{law_id:<12} sentences {len(b):>4}/{len(a):<4} "
              f"{'MATCH' if ok else f'DIFF ({n_diff} sentence diffs, stats equal={same_stats})'}",
              flush=True)
    json.dump(out, open(HERE / "reproduced_corpus.json", "w"), indent=1, ensure_ascii=False)
    print("ALL MATCH" if all_ok else "SOME DIFF")


def run(manifest_path):
    nlp = load_nlp()
    matcher = ns["build_deontic_matcher"](nlp.vocab)
    manifest = json.load(open(manifest_path))
    out_path = HERE / "extra_corpus.json"
    out = json.load(open(out_path)) if out_path.exists() else {}
    for law_id, spec in manifest.items():
        text = (REPO / spec["file"]).read_text(encoding="utf-8")
        ns["METADATA"][law_id] = spec["metadata"]
        r = ns["process_law"](law_id, text, nlp, matcher, min_tokens=MIN_TOKENS)
        out[law_id] = r
        st = r["stats"]
        print(f"{law_id:<16} words={st['word_count']:>6} sents={st['total_sentences']:>4} "
              f"oblig={st['obligation_sentences']:>4} defn={st['definitional_sentences']:>3} "
              f"modals={st['deontic_counts']}", flush=True)
        json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    {"verify": lambda: verify(), "run": lambda: run(sys.argv[2])}[sys.argv[1]]()
