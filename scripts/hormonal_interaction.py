"""
scripts/hormonal_interaction.py
================================
Concomitant medication interaction analysis for Dr. Nahman-Averbuch's question:
Is the valproate menstrual disorder signal stronger when hormonal medications
are also listed as concomitant drugs?

Splits valproate + menstrual disorder cases into:
  - hormonal_yes: case also lists a hormonal contraceptive or hormone therapy
  - hormonal_no:  case does NOT list any hormonal medication

Computes ROR and IC025 for each subgroup against the full FAERS background.

Usage (from sigvigil/ project root):
    python scripts/hormonal_interaction.py --processed-dir data/processed --out-dir output
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hormonal medication keyword list
# Covers OCs, hormonal IUDs, patches, rings, injections, HRT, progestin-only
# ---------------------------------------------------------------------------
HORMONAL_KEYWORDS = [
    # Combined OCs
    "ethinyl estradiol", "levonorgestrel", "norethindrone", "desogestrel",
    "drospirenone", "norgestimate", "etonogestrel", "norgestrel",
    "gestodene", "dienogest", "nomegestrol",
    # Brand names (most common)
    "yaz", "yasmin", "ortho tri-cyclen", "ortho tricyclen", "trinessa",
    "microgestin", "loestrin", "lo loestrin", "sprintec", "junel",
    "aviane", "lessina", "lutera", "sronyx", "camrese", "seasonique",
    "jolessa", "quartette", "amethia", "introvale",
    # Progestin-only / implant / IUD
    "nexplanon", "implanon", "mirena", "kyleena", "liletta", "skyla",
    "depo-provera", "depo provera", "medroxyprogesterone", "norplant",
    "camila", "heather", "errin", "jolivette", "nora-be", "lyza",
    # Patches and rings
    "xulane", "zafemy", "nuvaring", "annovera",
    # HRT / estrogen
    "estradiol", "estrogen", "premarin", "prempro", "provera",
    "prometrium", "progesterone", "vivelle", "climara", "estrogel",
    # Generic terms
    "oral contraceptive", "birth control", "contraceptive",
    "hormone replacement", "hormonal contraception",
]


def is_hormonal(drug_name: str) -> bool:
    """Return True if drug_name matches any hormonal keyword."""
    dn = str(drug_name).lower()
    return any(kw in dn for kw in HORMONAL_KEYWORDS)


def compute_ror_simple(a, b, c, d):
    """
    2x2 contingency table ROR with Haldane-Anscombe correction.
    a = drug+event, b = drug+no event, c = no drug+event, d = no drug+no event
    Returns (ror, lo95, hi95)
    """
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    ror = (a * d) / (b * c)
    log_se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo = np.exp(np.log(ror) - 1.96 * log_se)
    hi = np.exp(np.log(ror) + 1.96 * log_se)
    return ror, lo, hi


def compute_ic025_simple(a, n_drug, n_event, n_total):
    """
    IC025 (lower Bayesian credible bound of Information Component).
    Uses normal approximation: IC - 3.3 * sigma (Norén 2006 convention at alpha=0.025).
    """
    if a == 0 or n_total == 0:
        return float("-inf")
    e = (n_drug * n_event) / n_total
    ic = np.log2((a + 0.5) / (e + 0.5))
    # Variance approximation
    var = (1 / (a + 0.5)) - (1 / (n_drug + 0.5)) + (1 / (n_event + 0.5))
    ic025 = ic - 3.3 * np.sqrt(max(var, 0))
    return ic025


def main():
    parser = argparse.ArgumentParser(description="Hormonal co-medication interaction analysis")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load preprocessed data
    # ------------------------------------------------------------------
    log.info("Loading database from %s ...", args.processed_dir)
    p = Path(args.processed_dir)
    cases = pd.read_parquet(p / "cases.parquet")
    drugs_df = pd.read_parquet(p / "drugs.parquet")
    reactions_df = pd.read_parquet(p / "reactions.parquet")

    log.info("Loaded: %d cases, %d drug links, %d reaction links",
             len(cases), len(drugs_df), len(reactions_df))

    # ------------------------------------------------------------------
    # Step 1: Apply the standard adolescent female migraine filter
    # ------------------------------------------------------------------
    # Age filter
    if "age_years" in cases.columns:
        cases = cases[cases["age_years"].between(10, 21)].copy()
    # Sex filter
    if "sex" in cases.columns:
        cases = cases[cases["sex"].str.upper() == "F"].copy()

    target_caseids = set(cases["caseid"])
    drugs_target = drugs_df[drugs_df["caseid"].isin(target_caseids)].copy()
    reactions_target = reactions_df[reactions_df["caseid"].isin(target_caseids)].copy()

    log.info("After demographic filter: %d cases", len(cases))

    # ------------------------------------------------------------------
    # Step 2: Find cases where valproate is the drug
    # ------------------------------------------------------------------
    valproate_terms = ["valproate", "valproic acid", "divalproex", "depakote", "depakene"]

    def is_valproate(name):
        n = str(name).lower()
        return any(t in n for t in valproate_terms)

    valproate_cases = set(
        drugs_target[drugs_target["drug_name_normalized"].apply(is_valproate)]["caseid"]
    )
    log.info("Valproate cases in target cohort: %d", len(valproate_cases))

    # ------------------------------------------------------------------
    # Step 3: Find cases with menstrual disorder reaction
    # ------------------------------------------------------------------
    menstrual_terms = ["menstrual disorder", "menstruation irregular",
                       "oligomenorrhoea", "amenorrhoea", "dysmenorrhoea",
                       "menstruation delayed", "menstrual cycle irregular"]

    def is_menstrual(pt):
        t = str(pt).lower()
        return any(m in t for m in menstrual_terms)

    menstrual_cases = set(
        reactions_target[reactions_target["pt_name"].apply(is_menstrual)]["caseid"]
    )
    log.info("Menstrual disorder cases in target cohort: %d", len(menstrual_cases))

    # Cases with BOTH valproate AND menstrual disorder
    valproate_menstrual_cases = valproate_cases & menstrual_cases
    log.info("Valproate + menstrual disorder cases: %d", len(valproate_menstrual_cases))

    if len(valproate_menstrual_cases) == 0:
        log.error("No valproate + menstrual disorder cases found. Check data.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: For each valproate+menstrual case, check for hormonal co-meds
    # ------------------------------------------------------------------
    # Get ALL drugs for valproate+menstrual cases
    comed_drugs = drugs_target[drugs_target["caseid"].isin(valproate_menstrual_cases)].copy()
    comed_drugs["is_hormonal"] = comed_drugs["drug_name_normalized"].apply(is_hormonal)

    # Which caseids have at least one hormonal co-med?
    hormonal_caseids = set(
        comed_drugs[comed_drugs["is_hormonal"]]["caseid"]
    )
    non_hormonal_caseids = valproate_menstrual_cases - hormonal_caseids

    log.info("Valproate+menstrual WITH hormonal co-med: %d", len(hormonal_caseids))
    log.info("Valproate+menstrual WITHOUT hormonal co-med: %d", len(non_hormonal_caseids))

    # ------------------------------------------------------------------
    # Step 5: List the hormonal drugs detected
    # ------------------------------------------------------------------
    detected_hormonal = (
        comed_drugs[comed_drugs["is_hormonal"]][["caseid", "drug_name_normalized"]]
        .drop_duplicates()
        .sort_values("drug_name_normalized")
    )
    log.info("\nHormonal co-medications detected:")
    if len(detected_hormonal) > 0:
        for _, row in detected_hormonal.iterrows():
            log.info("  caseid %s: %s", row["caseid"], row["drug_name_normalized"])
    else:
        log.info("  (none found)")

    # ------------------------------------------------------------------
    # Step 6: Compute ROR and IC025 for each subgroup vs full FAERS background
    # ------------------------------------------------------------------
    N_total = len(cases)
    N_valproate = len(valproate_cases)

    # Count menstrual cases in full target cohort (not just valproate)
    all_menstrual_in_cohort = len(menstrual_cases)

    def compute_stats(drug_event_n, label):
        """drug_event_n = number of cases with BOTH drug AND event."""
        drug_only = N_valproate - drug_event_n      # valproate, no menstrual
        event_only = all_menstrual_in_cohort - drug_event_n  # menstrual, no valproate
        neither = N_total - drug_event_n - drug_only - max(event_only, 0)
        neither = max(neither, 0)

        ror, lo, hi = compute_ror_simple(
            drug_event_n, drug_only, max(event_only, 0), neither
        )
        ic025 = compute_ic025_simple(
            drug_event_n, N_valproate, all_menstrual_in_cohort, N_total
        )
        print(f"\n{'='*55}")
        print(f"  Subgroup: {label}")
        print(f"{'='*55}")
        print(f"  n (valproate + menstrual disorder): {drug_event_n}")
        print(f"  ROR:   {ror:.3f}  (95% CI: {lo:.3f} – {hi:.3f})")
        print(f"  IC025: {ic025:.3f}")
        print(f"  Signal (ROR lower CI > 1, n >= 3): "
              f"{'YES' if lo > 1 and drug_event_n >= 3 else 'NO'}")
        print(f"  Signal (IC025 > 0, n >= 3):         "
              f"{'YES' if ic025 > 0 and drug_event_n >= 3 else 'NO'}")
        return {
            "subgroup": label,
            "n_de": drug_event_n,
            "ror": round(ror, 3),
            "ror_lo95": round(lo, 3),
            "ror_hi95": round(hi, 3),
            "ror_signal": lo > 1 and drug_event_n >= 3,
            "ic025": round(ic025, 3),
            "ic025_signal": ic025 > 0 and drug_event_n >= 3,
        }

    print("\n" + "="*55)
    print("VALPROATE + MENSTRUAL DISORDER")
    print("Hormonal co-medication interaction analysis")
    print(f"Target cohort: adolescent females 10-21, FAERS 2004-2024")
    print(f"N total cohort: {N_total}")
    print(f"N valproate cases: {N_valproate}")
    print(f"N menstrual disorder cases (any drug): {all_menstrual_in_cohort}")
    print("="*55)

    results = []

    # Full group
    results.append(compute_stats(len(valproate_menstrual_cases), "ALL valproate+menstrual cases"))

    # With hormonal co-med
    results.append(compute_stats(len(hormonal_caseids), "WITH hormonal co-medication"))

    # Without hormonal co-med
    results.append(compute_stats(len(non_hormonal_caseids), "WITHOUT hormonal co-medication"))

    # ------------------------------------------------------------------
    # Step 7: Also run for topiramate + cognitive disorder as a comparison
    # ------------------------------------------------------------------
    topi_terms = ["topiramate", "topamax"]
    def is_topi(name):
        n = str(name).lower()
        return any(t in n for t in topi_terms)

    topi_cases = set(
        drugs_target[drugs_target["drug_name_normalized"].apply(is_topi)]["caseid"]
    )
    cog_terms = ["cognitive disorder", "cognitive impairment", "memory impairment",
                 "mental impairment", "thinking abnormal", "confusional state"]
    def is_cognitive(pt):
        t = str(pt).lower()
        return any(c in t for c in cog_terms)

    cog_cases = set(
        reactions_target[reactions_target["pt_name"].apply(is_cognitive)]["caseid"]
    )
    topi_cog_cases = topi_cases & cog_cases

    N_topi = len(topi_cases)
    all_cog = len(cog_cases)

    print("\n" + "="*55)
    print("TOPIRAMATE + COGNITIVE DISORDER (reference signal)")
    print("="*55)

    topi_drug_only = N_topi - len(topi_cog_cases)
    topi_event_only = all_cog - len(topi_cog_cases)
    topi_neither = N_total - len(topi_cog_cases) - topi_drug_only - max(topi_event_only, 0)
    topi_ror, topi_lo, topi_hi = compute_ror_simple(
        len(topi_cog_cases), topi_drug_only, max(topi_event_only, 0), max(topi_neither, 0)
    )
    topi_ic025 = compute_ic025_simple(len(topi_cog_cases), N_topi, all_cog, N_total)
    print(f"  n: {len(topi_cog_cases)}")
    print(f"  ROR: {topi_ror:.3f}  (95% CI: {topi_lo:.3f} – {topi_hi:.3f})")
    print(f"  IC025: {topi_ic025:.3f}")

    # ------------------------------------------------------------------
    # Step 8: Save results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    out_path = out_dir / "hormonal_interaction.csv"
    results_df.to_csv(out_path, index=False)
    log.info("\nResults saved to %s", out_path)

    print("\n" + "="*55)
    print("INTERPRETATION GUIDE")
    print("="*55)
    print("""
  If signal is STRONGER in the WITHOUT-hormonal group:
    -> The valproate menstrual signal is NOT explained by hormonal
       co-medication confounding. The drug itself is the driver.
       This STRENGTHENS the finding and supports Dr. Nahman-Averbuch's
       HPG axis disruption hypothesis.

  If signal is STRONGER in the WITH-hormonal group:
    -> Hormonal co-medications are interacting with valproate to amplify
       menstrual disruption. This supports her interaction hypothesis and
       suggests a drug-drug interaction mechanism worth investigating.

  If both groups show similar signals:
    -> Co-medication status doesn't modify the effect. Valproate's
       menstrual disruption signal is robust to hormonal exposure.
""")


if __name__ == "__main__":
    main()
