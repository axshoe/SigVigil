"""
examples/migraine_analysis.py
==============================
Worked example: full migraine preventive pharmacovigilance analysis.

Run this after preprocessing FAERS data (see SETUP.md).
Reproduces all figures and the HTML report for the SigVigil project.

Usage:
    python migraine_analysis.py --data-dir /data/sigvigil_processed/ \
                                 --output-dir ./sigvigil_output/
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DRUGS = [
    "topiramate", "valproate", "amitriptyline",
    "propranolol", "candesartan",
    "erenumab", "fremanezumab", "galcanezumab",
]

# Pre-specified events (documented before any analysis)
EVENTS = [
    # Metabolic
    "weight decreased", "decreased appetite", "anorexia",
    # Neurological
    "paraesthesia", "memory impairment", "cognitive disorder", "somnolence",
    # Psychiatric
    "depression", "anxiety", "suicidal ideation",
    # Reproductive
    "amenorrhoea", "dysmenorrhoea", "menstrual disorder",
    # Dermatological
    "alopecia", "rash",
    # Sleep
    "sleep disorder", "insomnia", "middle insomnia",
]


def run_analysis(data_dir: str, output_dir: str):
    import sigvigil as ps
    from sigvigil.analysis.stratified import run_stratified_comparison
    from sigvigil.analysis.sensitivity import run_sensitivity_suite

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FAERS database from %s", data_dir)
    db = ps.FAERSDatabase.from_directory(data_dir)

    # Primary analysis filter: adolescent females with migraine
    target_filter = ps.Filter(
        age=(10, 21),
        sex="F",
        indication="migraine",
    )

    logger.info("Running primary disproportionality analysis...")
    signals = db.analyze(
        drugs=DRUGS,
        events=EVENTS,
        population=target_filter,
        run_ebgm=True,
        correction="all",
    )

    # Save raw results
    signals.to_csv(str(out / "signals.csv"))
    logger.info("Signals saved to %s/signals.csv", output_dir)

    # Print summary
    summary = signals.summary(min_signals=2)
    logger.info("Signals detected by ≥2 methods:\n%s", summary[
        ["drug", "event", "n_de", "ror", "ic025", "ebgm", "n_methods_flagged"]
    ].to_string(index=False))

    # --- Figures ---
    logger.info("Generating figures...")

    # Figure 1: Heatmap
    signals.plot_heatmap(output_path=str(out / "fig1_heatmap.png"))
    logger.info("Fig 1 saved.")

    # Figure 2: Volcano
    signals.plot_volcano(output_path=str(out / "fig2_volcano.png"))
    logger.info("Fig 2 saved.")

    # Figure 3: Amplified signals (requires stratified comparison)
    logger.info("Running stratified comparison (target vs full FAERS)...")
    stratified = run_stratified_comparison(
        db=db,
        drugs=DRUGS,
        events=EVENTS,
        target_filter=target_filter,
        comparison_filter=None,  # full FAERS background
    )
    stratified.to_csv(str(out / "stratified.csv"), index=False)

    from sigvigil.viz.timeline import plot_amplified_signals
    plot_amplified_signals(stratified, output_path=str(out / "fig3_amplified.png"))
    logger.info("Fig 3 saved.")

    # Figure 4: Timeline
    signals.plot_timeline(db, population=target_filter,
                           output_path=str(out / "fig4_timeline.png"))
    logger.info("Fig 4 saved.")

    # Figure 5: EBGM vs ROR for topiramate
    signals.plot_ebgm_vs_ror(
        db=db, drug="topiramate",
        output_path=str(out / "fig5_ebgm_vs_ror.png")
    )
    logger.info("Fig 5 saved.")

    # --- Sensitivity analyses ---
    logger.info("Running sensitivity analyses...")
    sensitivity_results = run_sensitivity_suite(
        db=db,
        drugs=DRUGS,
        events=EVENTS,
        base_filter=target_filter,
    )
    sensitivity_results.to_csv(str(out / "sensitivity.csv"), index=False)

    # --- HTML report ---
    logger.info("Generating HTML report...")
    signals.generate_report(str(out / "sigvigil_report.html"))
    logger.info("Report saved to %s/sigvigil_report.html", output_dir)

    logger.info("=== Analysis complete. All outputs in %s ===", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sigvigil migraine analysis")
    parser.add_argument("--data-dir", default="/data/sigvigil_processed/",
                        help="Preprocessed FAERS data directory")
    parser.add_argument("--output-dir", default="./sigvigil_output/",
                        help="Output directory for figures and report")
    args = parser.parse_args()

    run_analysis(args.data_dir, args.output_dir)
