"""
scripts/preprocess.py
Run FAERS preprocessing from the terminal.

Usage (from the sigvigil/ project root):
    python scripts/preprocess.py --raw-dir data/faers_raw --out-dir data/processed

This reads the raw FAERS quarterly files you've already downloaded,
cleans and deduplicates them, normalizes drug names, and saves three
parquet files to data/processed/ that the analysis scripts use.
"""
import argparse, sys, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw FAERS data for SigVigil")
    parser.add_argument("--raw-dir", required=True, help="Folder containing your FAERS quarterly subfolders (e.g. 2020q1/, 2020q2/, ...)")
    parser.add_argument("--out-dir", default="data/processed", help="Where to write the cleaned parquet files (default: data/processed)")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year",   type=int, default=2024)
    parser.add_argument("--fuzzy-threshold", type=int, default=85, help="Drug name fuzzy match threshold 0-100 (default: 85)")
    args = parser.parse_args()

    from sigvigil.data.preprocessor import FAERSPreprocessor
    pp = FAERSPreprocessor(raw_data_dir=args.raw_dir, output_dir=args.out_dir)
    pp.run(start_year=args.start_year, end_year=args.end_year, fuzzy_threshold=args.fuzzy_threshold)
    print(f"\nDone. Processed files written to: {args.out_dir}")

if __name__ == "__main__":
    main()
