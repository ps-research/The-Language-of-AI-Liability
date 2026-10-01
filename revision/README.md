# Revision (October 2026)

Code and data for the revised analysis.

## What changed

- **Technique 2 (`t2/`).** HHI is computed over the three actor categories. Actors are matched on the subject's head noun phrase with whole words. Defined terms take the category of the seed term that heads them. The seed list has 50 terms.
- **Appendix D tables (`t3/`).** Word-list frequencies are recounted with the patterns the metrics use, counting each matched span once.
- **Non-AI baselines and a second before/after pair (`corpus_ext/`, `metrics/`).** One non-AI technology law per jurisdiction, and Colorado SB 24-205 against its successor SB 26-189.
- **Preprocessing (`preprocess/`).** Notebook 1, which builds `Data/processed/preprocessed_corpus.json`, and a script that re-runs it.

## Reproduce

Python 3.10, spaCy 3.8, `en_core_web_trf` 3.8.0. A GPU is optional. Run from the repository root.

1. `python revision/preprocess/preprocess.py verify` re-runs Notebook 1 on the 10 laws and compares the result with `Data/processed/preprocessed_corpus.json`.
2. `python revision/t2/reproduce_v1.py` re-runs the published Technique 2 cells and compares the result with `Data/processed/technique2_responsibility.json`.
3. `python revision/t2/t2_v2.py`, then `python revision/t2/post_v2.py`: corrected Technique 2, its statistics, and Figure 3b.
4. `python revision/t3/wordlist_v2.py`: Appendix D frequency tables.
5. `python revision/preprocess/preprocess.py run revision/corpus_ext/manifest.json`: preprocessing of the new texts.
6. `python revision/metrics/run_all.py`, then `python revision/metrics/baseline_tests.py`: Techniques 2-4 on the new texts, the baseline tests, and the Colorado pair.

`t2/v1_v2_reclassification.csv` lists every active-voice obligation sentence with its category before and after the correction.

## New texts (`corpus_ext/texts/`)

| File | Law | Source |
|---|---|---|
| `eu_dsa.txt` | Digital Services Act, Reg. (EU) 2022/2065, Articles only | EU Publications Office, CELEX 32022R2065 |
| `cn_pipl.txt` | Personal Information Protection Law (2021) | China Law Translate |
| `kr_pipa.txt` | Personal Information Protection Act, Act No. 19234 (2023) | Korea Legislation Research Institute |
| `co_sb21_190.txt` | Colorado Privacy Act, SB 21-190, signed act | Colorado General Assembly |
| `co_sb26_189.txt` | Colorado SB 26-189, signed act (2026) | Colorado General Assembly |
| `cn_algorithms.txt`, `cn_deep_synthesis.txt`, `cn_ai_labeling.txt` | Chinese AI rules (2022, 2023, 2025) | China Law Translate |

The three Chinese AI rules are processed but not used in the paper's analyses. `corpus_ext/extract.py` turns the downloaded source pages into these texts.
