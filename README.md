# SigVigil

**Systematic pharmacovigilance signal analysis of migraine preventives in adolescent females.**

SigVigil is the third project in the [Migraine Stratification Outcomes Framework (MSOF)](https://thexiulab.org) at The Xiu Lab. MSOF is a four-project computational pipeline spanning genetic variant annotation (ChanVar), phenotypic clustering (TraitStrata), pharmacovigilance signal detection (SigVigil), and adverse effect trajectory modeling (NeuroTrack).

SigVigil implements ROR, IC, PRR, and EBGM from first principles — no proprietary pharmacovigilance software, no black-box wrappers — and adds a stratified comparison layer that quantifies how adverse event signals differ in adolescent females relative to the general FAERS population on the same drugs. The first open-source, reproducible, multi-drug comparative pharmacovigilance analysis for this population.

---

## MSOF Pipeline Position

```
ChanVar ──► TraitStrata ──► SigVigil ──► NeuroTrack
 (variants)   (subgroups)  (signals)    (trajectories)
```

SigVigil provides the validated signal dataset that NeuroTrack uses for trajectory modeling.

---

## Installation

```bash
git clone https://github.com/axshoe/sigvigil
cd sigvigil
pip install -e .
```

## Quickstart

```python
import sigvigil as sv

db = sv.FAERSDatabase.from_directory('data/processed/')

signals = db.analyze(
    drugs=['topiramate', 'valproate', 'amitriptyline',
           'propranolol', 'candesartan', 'erenumab'],
    events=['weight decreased', 'alopecia', 'sleep disorder',
            'amenorrhoea', 'depression', 'paraesthesia'],
    population=sv.Filter(age=(10, 21), sex='F', indication='migraine'),
)

signals.summary()
signals.plot_heatmap(output_path='fig1_heatmap.png')
signals.generate_report('sigvigil_report.html')
```

---

## Statistical Methods

All implemented from scratch in `sigvigil/stats/`. Inline formula documentation with primary source citation.

| Method | Source | Signal criterion |
|--------|--------|-----------------|
| **ROR** (Reporting Odds Ratio) | Evans et al. 2001 | Lower 95% CI > 1, n ≥ 3 |
| **IC** (Information Component) | Norén et al. 2006 | IC025 > 0, n ≥ 3 |
| **PRR** (Proportional Reporting Ratio) | Evans 2001 | PRR ≥ 2, χ² ≥ 4, n ≥ 3 |
| **EBGM** (Empirical Bayes Geometric Mean) | DuMouchel 1999 | EB05 ≥ 2 |
| **Multiple testing** | Bonferroni, BH, Storey 2002 | α = 0.05 |

The novel contribution: stratified comparison of IC025 between the adolescent female migraine cohort and the full FAERS population on the same drugs (amplification delta).

---

## Repository Structure

```
sigvigil/
├── sigvigil/
│   ├── core.py              # FAERSDatabase, Filter, SignalResult
│   ├── data/
│   │   ├── downloader.py    # FAERS quarterly file download
│   │   ├── parser.py        # ASCII pipe-delimited file parser
│   │   ├── deduplicator.py  # Case deduplication
│   │   ├── normalizer.py    # Drug name normalization (fuzzy match)
│   │   └── preprocessor.py  # End-to-end pipeline
│   ├── stats/
│   │   ├── contingency.py   # 2×2 table builder
│   │   ├── ror.py           # Reporting Odds Ratio (Evans 2001)
│   │   ├── ic.py            # Information Component (Norén 2006)
│   │   ├── prr.py           # Proportional Reporting Ratio
│   │   ├── ebgm.py          # Empirical Bayes (DuMouchel 1999)
│   │   └── corrections.py   # Bonferroni, BH, Storey q-value
│   ├── analysis/
│   │   ├── stratified.py    # Adolescent female vs. general FAERS
│   │   └── sensitivity.py   # Five sensitivity analysis variants
│   ├── viz/
│   │   ├── heatmap.py       # Fig 1: Signal landscape heatmap
│   │   ├── scatter.py       # Fig 2: Volcano + Fig 5: EBGM vs ROR
│   │   └── timeline.py      # Fig 3: Amplified signals + Fig 4: Timeline
│   └── report/
│       └── generator.py     # Self-contained HTML report
├── tests/
│   └── test_stats.py        # Known-value unit tests (29 passing)
├── examples/
│   └── migraine_analysis.py # Full worked example
├── scripts/
│   ├── preprocess.py        # Run FAERS preprocessing from terminal
│   └── run_analysis.py      # Run full analysis from terminal
├── SETUP.md                 # Step-by-step setup for Windows + PyCharm
├── requirements.txt
└── README.md
```

---

## Tests

```bash
python -m pytest tests/ -v
# 29 passed
```

---

## Limitations

FAERS is subject to under-reporting, notoriety bias, Weber effect, confounding by indication, and absence of denominator data. Disproportionality signals indicate over-reporting relative to the database background — not causation. All signals require clinical judgment.

---

## Part of MSOF

- **ChanVar** — `github.com/axshoe/chanvar` — CACNA1A variant annotation
- **TraitStrata** — `github.com/axshoe/traitstrata` — Phenotypic clustering
- **SigVigil** — `github.com/axshoe/sigvigil` — Pharmacovigilance signals ← you are here
- **NeuroTrack** — `github.com/axshoe/neurotrack` — Adverse effect trajectories

*A. Xiu | The Xiu Lab | thexiulab.org*
