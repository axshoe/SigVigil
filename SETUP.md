# SETUP.md — SigVigil
**Part of the Migraine Stratification Outcomes Framework (MSOF)**
*A. Xiu | The Xiu Lab | thexiulab.org | github.com/axshoe/sigvigil*

---

## What You Will End Up With

By the end of this guide you will have SigVigil installed in PyCharm, your FAERS data preprocessed, all five publication figures generated, a self-contained HTML analysis report, and the project pushed to GitHub.

Every command block in this guide runs in the **PyCharm Terminal** (the Terminal tab at the very bottom of the PyCharm window), unless stated otherwise.

---

## Overview of the Workflow

```
Step 1: Open the project in PyCharm
Step 2: Create a virtual environment and install dependencies
Step 3: Organize your FAERS data folder
Step 4: Run preprocessing  (one time, ~30–90 min)
Step 5: Run the full analysis  (~5–15 min)
Step 6: Run tests to verify
Step 7: Push to GitHub
```

Total hands-on time: about 30 minutes of your time, plus waiting.

---

## Step 1: Open the Project in PyCharm

1. Unzip `sigvigil_project.zip` somewhere sensible, e.g. `C:\Users\Angie\code\sigvigil\`
2. In PyCharm: **File → Open** → select that `sigvigil` folder → **Trust Project**

---

## Step 2: Set Up a Virtual Environment

In the **PyCharm Terminal**, run these commands one at a time:

```
python -m venv .venv
```

Then activate it:
```
.venv\Scripts\activate
```

You should see `(.venv)` appear at the start of your terminal line. If Windows blocks the second command, run this first, then try again:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Now install everything:
```
pip install -e .
```

This installs all dependencies and the `sigvigil` package itself. It reads from `requirements.txt` automatically.

**Verify it worked:**
```
python -c "import sigvigil; print('sigvigil OK')"
```

Expected output: `sigvigil OK`

**Tell PyCharm to use this interpreter:** Bottom-right of the PyCharm window → click the Python version shown → **Add New Interpreter → Add Local Interpreter → Existing** → navigate to `.venv\Scripts\python.exe` → OK.

---

## Step 3: Organize Your FAERS Data

Your FAERS quarterly folders are already downloaded. Create this folder structure inside the project (you can do this in PyCharm's file panel on the left, or in Windows Explorer):

```
sigvigil/                     ← your PyCharm project root
├── data/
│   ├── faers_raw/            ← put your FAERS folders here
│   │     ├── 2020q1/         ← each should contain .txt files
│   │     ├── 2020q2/
│   │     └── ...
│   └── processed/            ← SigVigil creates this automatically
├── output/                   ← SigVigil creates this automatically
```

Move or copy your FAERS quarterly folders (named like `2020q1`, `2021q2`, etc.) into `data/faers_raw/`. Each quarterly folder should contain files like `DEMO20Q1.txt`, `DRUG20Q1.txt`, `REAC20Q1.txt`, etc.

---

## Step 4: Run Preprocessing

In the **PyCharm Terminal** (with `.venv` active):

```
python scripts/preprocess.py --raw-dir data/faers_raw --out-dir data/processed
```

This reads all your raw FAERS files, deduplicates cases, normalizes drug names, and saves three clean files to `data/processed/`. You only run this once.

You'll see progress lines printing as it works through each quarterly file. When it finishes:
```
Done. Processed files written to: data/processed
```

Three files will be in `data/processed/`: `cases.parquet`, `drugs.parquet`, `reactions.parquet`.

**Optional flags:**
- Only process a specific year range: `--start-year 2015 --end-year 2024`
- Looser drug name matching: `--fuzzy-threshold 80` (default is 85; lower = more permissive)

---

## Step 5: Run the Full Analysis

```
python scripts/run_analysis.py --processed-dir data/processed --out-dir output
```

This loads the preprocessed data, runs all four disproportionality methods, runs the stratified comparison (adolescent female cohort vs. full FAERS), runs sensitivity analyses, generates all five figures as PNGs, and produces the HTML report.

When finished, `output/` will contain:

```
output/
├── fig1_heatmap.png        ← signal landscape heatmap
├── fig2_volcano.png        ← volcano plot
├── fig3_amplified.png      ← adolescent-amplified signals bar chart
├── fig4_timeline.png       ← reporting rate over time
├── fig5_ebgm_ror.png       ← EBGM vs ROR (Bayesian shrinkage demo)
├── signals.csv             ← all drug-event results
├── stratified.csv          ← target vs. background comparison
├── sensitivity.csv         ← sensitivity analysis results
└── sigvigil_report.html    ← open this in any browser
```

If EBGM is taking too long, add `--no-ebgm` to skip the Bayesian step:
```
python scripts/run_analysis.py --processed-dir data/processed --out-dir output --no-ebgm
```

---

## Step 6: Run the Tests

```
python -m pytest tests/ -v
```

Expected: 29 tests all `PASSED`. If any fail, run `pip install -r requirements.txt` and retry.

---

## Step 7: Figure Captions

The five figures go on the SigVigil lab writeup page. Standard captions:

**Fig. 1** — Signal landscape heatmap. Rows: adverse event categories (metabolic, neurological, psychiatric, reproductive, dermatological, sleep). Columns: migraine preventive drugs. Color = IC025 (lower Bayesian credible bound); blue = below null, red = strong signal. Cell labels show ROR where lower 95% CI > 1 and n ≥ 3.

**Fig. 2** — Volcano plot of all drug-event pairs tested in the adolescent female migraine cohort (FAERS 2004–2024). X-axis: log₂(ROR); Y-axis: −log₁₀(p-value). Point size ∝ √N. Color indicates drug. Points above the horizontal line (p = 0.05) and right of the vertical (ROR = 2) are labeled.

**Fig. 3** — Adverse events with amplified signals in adolescent females relative to the general FAERS population on the same drug. IC025 amplification delta = IC025(adolescent female cohort) − IC025(full FAERS). Only pairs with positive target IC025 shown.

**Fig. 4** — Annual FAERS report counts by drug for adolescent females with migraine, 2004–2024. Dotted vertical = CGRP monoclonal antibody FDA approval year (2018). Counts reflect deduplicated case-level data.

**Fig. 5** — EBGM vs. ROR for topiramate across all tested adverse events. Points below the identity line are shrunk toward null by the Bayesian prior (small-N shrinkage). Point size and color ∝ N.

---

## Step 8: Push to GitHub

**One-time Git setup** (if you haven't done this before):
```
git config --global user.name "Angie Xiu"
git config --global user.email "your@email.com"
```

**Create the GitHub repo:**
1. Go to github.com → **New repository** → name it `sigvigil` → set to **Public** → do **not** add a README → **Create repository**

**Create a .gitignore** so your multi-GB data files don't get pushed. Create a file called `.gitignore` in the project root (right-click in PyCharm's file panel → New → File → name it `.gitignore`) with this content:
```
data/
output/
.venv/
*.egg-info/
__pycache__/
*.pyc
.pytest_cache/
*.parquet
```

**Push everything:**
```
git init
git add .
git commit -m "Initial commit: SigVigil v1.0.0 — MSOF project 3"
git branch -M main
git remote add origin https://github.com/axshoe/sigvigil.git
git push -u origin main
```

GitHub will ask for your username and a **Personal Access Token** (not your password). To get one: GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic) → Generate new token → check the `repo` box → copy the token → paste it when the terminal asks for a password.

---

## Quick Reference: The Three Commands You'll Actually Use

```
# Every time you open a new terminal session:
.venv\Scripts\activate

# Once, after organizing your data:
python scripts/preprocess.py --raw-dir data/faers_raw --out-dir data/processed

# Whenever you want to generate/refresh results:
python scripts/run_analysis.py --processed-dir data/processed --out-dir output
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'sigvigil'"**
Your virtual environment is not active. Run `.venv\Scripts\activate`.

**"No cases file found in data/processed"**
Preprocessing hasn't run yet, or didn't finish. Re-run `python scripts/preprocess.py ...`.

**"python is not recognized"**
PyCharm is using the wrong interpreter. Bottom-right of PyCharm → select `.venv\Scripts\python.exe`.

**Drug match rate in the logs is very low**
Add `--fuzzy-threshold 75` to the preprocess command.

**Git push fails asking for a password**
GitHub doesn't accept account passwords anymore. Use a Personal Access Token (see Step 8).

---

*SigVigil — MSOF Project 3 of 4*
*A. Xiu | The Xiu Lab | thexiulab.org*
