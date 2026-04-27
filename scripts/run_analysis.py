"""
scripts/run_analysis.py
Run the full SigVigil analysis and generate all figures + report.

Usage (from the sigvigil/ project root):
    python scripts/run_analysis.py --processed-dir data/processed --out-dir output

Requires: preprocessed data in --processed-dir (run preprocess.py first).
"""
import argparse, logging, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DRUGS = ["topiramate","valproate","amitriptyline","propranolol","candesartan",
         "erenumab","fremanezumab","galcanezumab"]

EVENTS = [
    "weight decreased","decreased appetite","anorexia",
    "paraesthesia","memory impairment","cognitive disorder","somnolence",
    "depression","anxiety","suicidal ideation",
    "amenorrhoea","dysmenorrhoea","menstrual disorder",
    "alopecia","rash",
    "sleep disorder","insomnia","middle insomnia",
]

def main():
    parser = argparse.ArgumentParser(description="Run SigVigil analysis")
    parser.add_argument("--processed-dir", default="data/processed", help="Preprocessed FAERS data directory")
    parser.add_argument("--out-dir", default="output", help="Where to write figures, CSVs, and the HTML report")
    parser.add_argument("--no-ebgm", action="store_true", help="Skip EBGM (faster, omits Bayesian estimates)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import sigvigil as sv
    from sigvigil.analysis.stratified import run_stratified_comparison
    from sigvigil.analysis.sensitivity import run_sensitivity_suite
    from sigvigil.viz.timeline import plot_amplified_signals

    log.info("Loading database from %s ...", args.processed_dir)
    db = sv.FAERSDatabase.from_directory(args.processed_dir)

    filt = sv.Filter(age=(10,21), sex="F", indication="migraine")

    log.info("Running disproportionality analysis (%d drugs × %d events) ...", len(DRUGS), len(EVENTS))
    signals = db.analyze(drugs=DRUGS, events=EVENTS, population=filt,
                         run_ebgm=not args.no_ebgm, correction="all")

    signals.to_csv(f"{args.out_dir}/signals.csv")
    log.info("Signal results: %s/signals.csv", args.out_dir)

    log.info("Running stratified comparison ...")
    strat = run_stratified_comparison(db, DRUGS, EVENTS, filt)
    strat.to_csv(f"{args.out_dir}/stratified.csv", index=False)

    log.info("Running sensitivity analyses ...")
    sens = run_sensitivity_suite(db, DRUGS, EVENTS, filt)
    sens.to_csv(f"{args.out_dir}/sensitivity.csv", index=False)

    log.info("Generating figures ...")
    signals.plot_heatmap(output_path=f"{args.out_dir}/fig1_heatmap.png")
    signals.plot_volcano(output_path=f"{args.out_dir}/fig2_volcano.png")
    plot_amplified_signals(strat, output_path=f"{args.out_dir}/fig3_amplified.png")
    # FIX 1: removed population=filt (it's already stored in signals)
    signals.plot_timeline(db, output_path=f"{args.out_dir}/fig4_timeline.png")
    # FIX 2: removed db=db (method does not accept it)
    signals.plot_ebgm_vs_ror(drug="topiramate", output_path=f"{args.out_dir}/fig5_ebgm_ror.png")

    log.info("Generating HTML report ...")
    signals.generate_report(f"{args.out_dir}/sigvigil_report.html")

    print(f"\n=== Done. All outputs written to: {args.out_dir}/ ===")
    print(f"  figures:  fig1–fig5 .png")
    print(f"  tables:   signals.csv, stratified.csv, sensitivity.csv")
    print(f"  report:   sigvigil_report.html")

if __name__ == "__main__":
    main()