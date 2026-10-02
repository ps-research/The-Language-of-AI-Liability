# The Language of AI Liability

Code and data for a computational linguistic analysis of obligation and responsibility in 10 AI governance frameworks from the EU, China, South Korea, and four US states.

## Pipeline

Run the notebooks from the repository root, in this order. They need Python 3.10, spaCy 3.8, and `en_core_web_trf` 3.8.0. A GPU is optional. Outputs go to `Data/processed/` and figures to `Data/figures/`.

| # | Notebook | What it does | Main outputs |
|---|---|---|---|
| 1 | `Data_Sanity_Checker_and_Fixer` | Checks the raw texts, fixes Colorado section markers, and separates the EU AI Act's binding text from its recitals | `sanity_check_results.json`, `eu_ai_act_binding.txt` |
| 2 | `Data_Preprocessing` | Sentence segmentation, deontic modal detection, and obligation sentences; the same pipeline for the non-AI baselines | `preprocessed_corpus.json`, `baselines_corpus.json` |
| 3 | `Deontic_Modal_Analysis` | Technique 1: deontic modal density and composition | Figures 1-2 |
| 4 | `Responsibility_Attribution_Analysis` | Technique 2: voice, actor categories, and concentration (HHI) | `technique2_responsibility.json`, `technique2_reclassification.csv` |
| 5 | `Obligation_Complexity_Analysis` | Technique 3: tree depth, conditional and cross-reference density | `technique3_complexity.json` |
| 6 | `Regulatory_Specificity_Index` | Technique 4: numeric, definitional, temporal, and hedge measures | `technique4_specificity.json` |
| 7 | `Word_List` | Word-list frequencies and the false-positive sample | `word_list_frequencies.json` |
| 8 | `Non_AI_Baselines` | Each AI law against a non-AI law from its jurisdiction, and Colorado SB 24-205 against SB 26-189 | `baseline_tests.json` |

## Data

**AI governance frameworks (main corpus).** EU AI Act (Regulation 2024/1689, binding provisions), China's Interim Measures for Generative AI Services (China Law Translate), South Korea's AI Basic Act, and California SB 1047, SB 53, SB 942, AB 2013, Colorado SB 24-205, Texas HB 149 (TRAIGA), and Illinois HB 3773 (state legislatures).

**Non-AI baselines and the Colorado successor.**

| File | Law | Source |
|---|---|---|
| `eu_dsa.txt` | Digital Services Act, Regulation 2022/2065, Articles only | EU Publications Office |
| `cn_pipl.txt` | Personal Information Protection Law (2021) | China Law Translate |
| `kr_pipa.txt` | Personal Information Protection Act, Act No. 19234 (2023) | Korea Legislation Research Institute |
| `co_sb21_190.txt` | Colorado Privacy Act, SB 21-190 | Colorado General Assembly |
| `co_sb26_189.txt` | Colorado SB 26-189 (2026), successor of SB 24-205 | Colorado General Assembly |

## October 2026 revision

- Technique 2 computes HHI over the three actor categories. Actors are matched on the subject's head noun with whole words, and defined terms take the category of the seed term that heads them. `technique2_reclassification.csv` lists every sentence whose category changed.
- Word-list frequencies are counted with the patterns the metrics use.
- Non-AI baselines and the Colorado SB 24-205 / SB 26-189 pair are new.
- The sanity-check and preprocessing notebooks are included, and all notebooks run from the repository.
