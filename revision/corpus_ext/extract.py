"""Turn the downloaded sources into plain-text law files, following the
conventions of the original corpus (code/Data/*.txt):

* China (China Law Translate, like china_genai_measures.txt): the law body from
  Chapter I / Article 1 to the end of the post text (closing lines included),
  without the site's citation/rating/related-post blocks.
* EU DSA (EU Publications Office XHTML): binding text only, from CHAPTER I to
  the signatures; recitals and footnotes dropped (the rule Notebook 0 used for
  the EU AI Act).
* Korea PIPA (KLRI official English, like the AI Basic Act text): Articles plus
  the addenda of the current act (Act No. 19234) only.
* Colorado signed acts (like co_sb205.txt): full signed-act text minus running
  page headers and the signature block, then Notebook 0's Colorado fix applied
  verbatim: re.sub(r'(SECTION)\\s*\\n\\s*(\\d+)', r'\\1 \\2', text).
"""
import json
import re
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "raw", HERE / "texts"
OUT.mkdir(exist_ok=True)
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
BLOCK = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "dd", "dt"}


def block_text(el):
    """Text with line breaks only at block elements (inline links stay joined)."""
    for br in el.find_all("br"):
        br.replace_with("\n")
    for b in el.find_all(BLOCK):
        b.insert_after("\n")
    t = el.get_text("")
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def clt(fname):
    soup = BeautifulSoup(open(RAW / fname, encoding="utf-8"), "lxml")
    pc = soup.select_one("div.post-content")
    end_classes = {"citepress-box", "yasr-auto-insert-visitor", "molongui-clearfix", "m-a-box", "yarpp"}
    lines = []
    for ch in pc.find_all(recursive=False):
        if ch.name == "div" and end_classes & set(ch.get("class", [])):
            break                      # citation, rating, author box, related posts
        if ch.name in ("div", "pre"):  # site boxes before the law (quicktake links, metadata)
            continue
        t = re.sub(r"\s+", " ", ch.get_text(" ", strip=True))
        if t and not CJK.search(t):
            lines.append(t)
    a1 = next(i for i, l in enumerate(lines) if re.match(r"Article 1\b", l))
    ch1 = [i for i in range(max(0, a1 - 2), a1) if re.match(r"Chapter I\b", lines[i])]
    return "\n\n".join(lines[ch1[0] if ch1 else a1:])


def dsa():
    soup = BeautifulSoup(open(RAW / "eu_dsa_cellar.xhtml", encoding="utf-8"), "lxml")
    for fn in soup.select(".note, .footnote, p.note"):
        fn.decompose()
    t = block_text(soup.body or soup)
    start = re.search(r"^CHAPTER I$", t, re.M).start()
    end = re.search(r"^Done at Strasbourg,.*$", t[start:], re.M)
    tail = t[start + end.end():]
    sig = re.match(r"(?s)(.*?For the Council\nThe President\n[^\n]+)", tail)
    return t[start:start + end.end()] + (sig.group(1) if sig else "")


def pipa():
    """KLRI renders the whole Act twice (div.JO ids 2199xxx and 2225xxx, same
    words); keep the first rendering, then the current act's addenda once."""
    soup = BeautifulSoup(open(RAW / "kr_pipa_klri.html", encoding="utf-8"), "lxml")
    jos = soup.select("div.JO")
    first = [j for j in jos if j.get("id", "")[:4] == jos[0].get("id", "")[:4]]
    arts = [block_text(BeautifulSoup(str(j), "lxml")).replace("법령보기 화면", "").strip() for j in first]
    articles = "\n\n".join(a for a in arts if a)
    t = block_text(soup.body)
    t = "\n".join(l for l in t.split("\n") if not CJK.search(l))
    t = re.sub(r"(?m)^(.+)\n\1$", r"\1", t)              # duplicated headings
    cur = t.index("ADDENDA <Act No. 19234, Mar. 14, 2023>")
    nxt = t.find("ADDENDA <", cur + 10)
    return articles + "\n\n" + t[cur:nxt if nxt != -1 else None].strip()


def colorado(fname, bill):
    doc = pymupdf.open(RAW / fname)
    t = "\n".join(p.get_text() for p in doc)
    t = re.sub(rf"(?m)^PAGE \d+-SENATE BILL {bill}\s*$\n?", "", t)
    lines = t.split("\n")
    pres = next(i for i, l in enumerate(lines) if l.strip().startswith("PRESIDENT OF"))
    # last line ending a sentence of running text (previous line is a full text line)
    cut = next(i for i in range(pres - 1, 0, -1)
               if lines[i].rstrip().endswith(".") and len(lines[i - 1].strip()) >= 40)
    t = "\n".join(lines[:cut + 1])
    t = re.sub(r'(SECTION)\s*\n\s*(\d+)', r'\1 \2', t)   # Notebook 0, verbatim
    return t.strip() + "\n"


JOBS = {
    "cn_algorithms":     lambda: clt("cn_algorithms.html"),
    "cn_deep_synthesis": lambda: clt("cn_deep_synthesis.html"),
    "cn_ai_labeling":    lambda: clt("cn_ai_labeling.html"),
    "cn_pipl":           lambda: clt("cn_pipl.html"),
    "eu_dsa":            dsa,
    "kr_pipa":           pipa,
    "co_sb26_189":       lambda: colorado("co_sb26_189_signed.pdf", "26-189"),
    "co_sb21_190":       lambda: colorado("co_sb21_190_signed.pdf", "21-190"),
}

if __name__ == "__main__":
    report = {}
    for law_id, fn in JOBS.items():
        text = fn()
        (OUT / f"{law_id}.txt").write_text(text, encoding="utf-8")
        arts = [int(n) for n in re.findall(r"(?m)^Article (\d+)\b", text)]
        report[law_id] = {
            "words": len(text.split()),
            "cjk_chars": len(CJK.findall(text)),
            "articles_found": len(arts),
            "article_max": max(arts) if arts else None,
            "head": text[:160].replace("\n", " | "),
            "tail": text[-220:].replace("\n", " | "),
        }
        r = report[law_id]
        print(f"\n### {law_id}: {r['words']} words, CJK={r['cjk_chars']}, "
              f"'Article N' line starts={r['articles_found']} (max {r['article_max']})")
        print("  HEAD:", r["head"]); print("  TAIL:", r["tail"])
    json.dump(report, open(HERE / "extract_report.json", "w"), indent=1)
