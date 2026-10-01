"""Technique 2 (responsibility attribution), corrected actor classification (v2).

Unchanged from the published notebook (Responsibility_Attribution_Analysis.ipynb):
  * voice classification (cell 4)       -> reused from reproduce_v1.py output
  * subject extraction, extract_subject (cell 6) -> copied verbatim below

Changed (the fixes for the October 2026 revision):
  1. HHI is computed over the three actor categories (regulated entity,
     government/institutional, generic), as the paper's method describes.
     The published code computed it over individual actor labels.
  2. Actors are classified by the subject's head noun phrase (head noun plus
     its compound/adjectival modifiers), matched on whole words. The published
     code matched substrings anywhere in the subject's full subtree, so e.g.
     "entity" matched "identity" and modifiers deep in the phrase could decide
     the category.
  3. Defined terms are no longer all labelled "regulated_entity": a defined term
     gets the category of the seed term that heads it (e.g. "frontier developer"
     -> developer -> regulated entity, "council" -> government); defined terms
     headed by no seed term (e.g. "covered model", "computing cluster") are not
     actors.
  4. The seed list is the union of the published code's SEED_ACTORS and the
     paper's Appendix D Table 9; where the two disagree (member state, notified
     body) Table 9's category (government/institutional) is used.
  5. Pronoun subjects are labelled "anaphoric" (the paper's text describes this;
     the published code put them under "unclassified").
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

import spacy

REPO = Path(__file__).resolve().parents[2]  # repository root
OUT = REPO / "revision" / "t2"

LAW_ORDER = [
    "eu_ai_act", "china_genai", "south_korea",
    "ca_sb1047", "ca_sb53", "ca_sb942", "ca_ab2013",
    "co_sb205", "tx_traiga", "il_hb3773",
]

# ── Seed list v2: union of code SEED_ACTORS and paper Table 9 (50 terms) ──
SEED_V2 = {
    "regulated_entity": [
        "developer", "provider", "deployer", "operator", "distributor",
        "importer", "manufacturer", "authorised representative", "employer",
        "covered entity", "service provider", "responsible person",
        "data controller", "processor", "vendor", "company", "business",
        "organization",
    ],
    "government": [
        "commission", "agency", "authority", "office", "board", "committee",
        "department", "ministry", "attorney general", "secretary", "government",
        "member state", "notified body", "market surveillance",
        "national authority", "competent authority", "advisory body",
        "council", "minister", "state",
    ],
    "generic": [
        "person", "entity", "individual", "one", "natural person",
        "legal person", "user", "applicant", "complainant", "affected person",
        "third party", "public",
    ],
}
TERM_CATEGORY = {t: c for c, ts in SEED_V2.items() for t in ts}
assert len(TERM_CATEGORY) == 50, len(TERM_CATEGORY)

# Spelling variants folded onto one seed term (not new terms).
VARIANTS = {"authorized": "authorised", "organisation": "organization",
            "governmental": "government"}
# "market surveillance" is a modifier in "market surveillance authority";
# a phrase containing a government modifier is government/institutional.
GOV_MODIFIERS = {"government", "market surveillance"}

# Anaphoric = the subject head is one of these pronouns (a fixed list; spaCy's
# PRON tag also fires on list markers such as "(a)", so it is not used).
PRONOUNS = {"it", "they", "he", "she", "which", "who", "that", "this",
            "these", "those", "we", "you", "i", "them", "its", "their"}


def normalise(words):
    return " ".join(VARIANTS.get(w, w) for w in words)


def singular(word):
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"          # bodies -> body, authorities -> authority
    if word.endswith("ses") or word.endswith("xes"):
        return word[:-2]                # businesses -> business
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]                # providers -> provider
    return word


def classify_phrase(words):
    """Classify a head noun phrase given as lower-case words, head noun last.

    Only seed terms that END with the head noun are candidates, so the head
    decides the actor type; the longest such term wins (e.g. "member state"
    over "state"). Returns (label, category) or (None, "unclassified").
    """
    if not words:
        return None, "unclassified"
    words = [w for w in words if re.fullmatch(r"[a-z][a-z\-']*", w)]
    if not words:
        return None, "unclassified"
    words = words[:-1] + [singular(words[-1])]
    phrase = normalise(words)
    best = None
    for term, cat in TERM_CATEGORY.items():
        t = term.split()
        if phrase.split()[-len(t):] == t and (best is None or len(t) > len(best[0].split())):
            best = (term, cat)
    if best is None:
        return None, "unclassified"
    term, cat = best
    if cat != "government" and any(f" {m} " in f" {phrase} " for m in GOV_MODIFIERS):
        return term, "government"
    return term, cat


# ── Verbatim from notebook cell 6 (unchanged subject extraction) ──
def extract_subject(doc):
    """Extract the subject noun phrase from a parsed sentence.
    Returns the text of the subject's subtree (the full noun phrase)
    and the head noun."""
    for token in doc:
        if token.dep_ == "nsubj":
            # Get the full noun phrase via the subtree
            subtree_span = doc[token.left_edge.i : token.right_edge.i + 1]
            return {
                "head": token.text.lower(),
                "full_np": subtree_span.text,
                "head_lemma": token.lemma_.lower(),
            }
    return None


def head_phrase(doc):
    """Head noun of the same nsubj that extract_subject uses, plus its
    compound/adjectival modifiers (in surface order)."""
    for token in doc:
        if token.dep_ == "nsubj":
            mods = []
            stack = [c for c in token.lefts if c.dep_ in ("compound", "amod")]
            while stack:
                c = stack.pop()
                mods.append(c)
                stack.extend(cc for cc in c.lefts if cc.dep_ in ("compound", "amod"))
            toks = sorted(mods, key=lambda t: t.i) + [token]
            return token, [t.text.lower() for t in toks]
    return None, []


def classify_words(words):
    """Classify a subject from its head phrase words (head noun last)."""
    if not words:
        return None, "unclassified"
    if len(words) == 1 and words[0] in PRONOUNS:
        return words[0], "anaphoric"
    return classify_phrase(words)


def defined_term_categories(results):
    """Stage 2 for transparency: every term each law defines, with the category
    its head noun gives it (None = not an actor)."""
    pat = re.compile(
        r'["“]([^"”]+)["”]\s+means\b'
        r'|\b(\w[\w\s]*?)\s+means\b'
        r'|["“]([^"”]+)["”]\s+has\s+the\s+(?:same\s+)?meaning',
        re.IGNORECASE)
    table = {}
    for law_id in LAW_ORDER:
        rows = {}
        for s in results[law_id]["sentences"]:
            if not s["is_definitional"]:
                continue
            for groups in pat.findall(s["text"]):
                term = next((g.strip().lower() for g in groups if g.strip()), None)
                if term and len(term) < 60:
                    label, cat = classify_phrase(term.split())
                    rows[term] = None if cat == "unclassified" else cat
        table[law_id] = rows
    return table


def main():
    results = json.load(open(REPO / "Data" / "processed" / "preprocessed_corpus.json"))
    voice = json.load(open(OUT / "v1_voice.json"))
    v1 = {(r["law_id"], r["sent_id"]): r for r in json.load(open(OUT / "v1_extractions.json"))}

    cache_path = OUT / "v2_parsed_subjects.json"
    if cache_path.exists():
        parsed = json.load(open(cache_path))
    else:
        nlp = spacy.load("en_core_web_trf")
        nlp.max_length = 700_000
        parsed = {}
        for law_id in LAW_ORDER:
            active = [v for v in voice if v["law_id"] == law_id and v["voice"] == "active"]
            for v, doc in zip(active, nlp.pipe([v["text"] for v in active], batch_size=16)):
                subj = extract_subject(doc)
                _, words = head_phrase(doc)
                parsed[f"{law_id}#{v['sent_id']}"] = {"full_np": (subj or {}).get("full_np"), "words": words}
            print(f"parsed {law_id}", flush=True)
        json.dump(parsed, open(cache_path, "w"), indent=1)

    real = ("regulated_entity", "government", "generic")
    per_law, diff_rows = {}, []
    for law_id in LAW_ORDER:
        active = [v for v in voice if v["law_id"] == law_id and v["voice"] == "active"]
        cats, labels = Counter(), Counter()
        for v in active:
            rec = parsed[f"{law_id}#{v['sent_id']}"]
            subj = {"full_np": rec["full_np"]} if rec["full_np"] is not None else None
            words = rec["words"]
            label, cat = classify_words(words)
            cats[cat] += 1
            if cat in real:
                labels[label] += 1
            old = v1[(law_id, v["sent_id"])]
            assert (old["subject"] or {}).get("full_np") == (subj or {}).get("full_np"), \
                f"subject extraction diverged at {law_id}#{v['sent_id']}"
            diff_rows.append({
                "law_id": law_id, "sent_id": v["sent_id"],
                "subject": (subj or {}).get("full_np", ""),
                "head_phrase": " ".join(words),
                "v1_label": old["v1_label"], "v1_category": old["v1_category"],
                "v2_label": label, "v2_category": cat,
                "changed": old["v1_category"] != cat,
            })
        n_real = sum(cats[c] for c in real)
        shares = [cats[c] / n_real for c in real] if n_real else []
        label_shares = [n / n_real for n in labels.values()] if n_real else []
        per_law[law_id] = {
            "total_active": len(active),
            "category_counts": {c: cats.get(c, 0) for c in (*real, "anaphoric", "unclassified")},
            "classified_real_actors": n_real,
            "coverage": n_real / len(active) if active else 0.0,
            "hhi_category": sum(s * s for s in shares),
            "hhi_label": sum(s * s for s in label_shares),
            "category_shares": dict(zip(real, shares)) if n_real else {},
            "top_labels": dict(labels.most_common(5)),
        }
        p = per_law[law_id]
        print(f"{law_id:<12} active={p['total_active']:>3} real={n_real:>3} "
              f"HHI_cat={p['hhi_category']:.3f} HHI_label={p['hhi_label']:.3f} "
              f"cats={p['category_counts']}", flush=True)

    json.dump(per_law, open(OUT / "technique2_responsibility_v2.json", "w"), indent=2)
    json.dump(defined_term_categories(results), open(OUT / "defined_terms_v2.json", "w"), indent=2)
    with open(OUT / "v1_v2_reclassification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diff_rows[0]))
        w.writeheader()
        w.writerows(diff_rows)
    print(f"changed categories: {sum(r['changed'] for r in diff_rows)} of {len(diff_rows)} active-voice sentences")


if __name__ == "__main__":
    main()
