"""
sigvigil.core
==============
Main database and analysis orchestration layer.

FAERSDatabase: loads the cleaned analysis dataset and dispatches
  to statistical modules.
Filter: declarative specification of population subsets.
SignalResult: structured container for analysis output.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sigvigil.stats.ror import compute_ror
from sigvigil.stats.ic import compute_ic
from sigvigil.stats.prr import compute_prr
from sigvigil.stats.ebgm import fit_ebgm_prior, compute_ebgm_posterior
from sigvigil.stats.corrections import bonferroni, benjamini_hochberg, storey_qvalue
from sigvigil.stats.contingency import build_contingency

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

@dataclass
class Filter:
    """Declarative population filter applied to a FAERSDatabase.

    Parameters
    ----------
    age : tuple of (min_age, max_age) in years, inclusive.
    sex : 'F', 'M', or None (no filter).
    indication : substring to match in indication field (case-insensitive).
    report_years : optional (start_year, end_year) inclusive.
    serious_only : if True, keep only reports with serious outcome.
    hcp_only : if True, keep only reports from healthcare providers.
    primary_suspect_only : if True, keep only cases where role_cod='PS'.
    """
    age: Optional[Tuple[float, float]] = None
    sex: Optional[str] = None
    indication: Optional[str] = None
    report_years: Optional[Tuple[int, int]] = None
    serious_only: bool = False
    hcp_only: bool = False
    primary_suspect_only: bool = False

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply this filter to a case DataFrame and return the filtered subset."""
        mask = pd.Series(True, index=df.index)

        if self.age is not None:
            lo, hi = self.age
            if "age_years" in df.columns:
                mask &= df["age_years"].between(lo, hi)

        if self.sex is not None:
            if "sex" in df.columns:
                mask &= df["sex"].str.upper() == self.sex.upper()

        if self.indication is not None:
            if "indication" in df.columns:
                mask &= df["indication"].str.contains(
                    self.indication, case=False, na=False
                )

        if self.report_years is not None:
            lo_yr, hi_yr = self.report_years
            if "report_year" in df.columns:
                mask &= df["report_year"].between(lo_yr, hi_yr)

        if self.serious_only and "serious" in df.columns:
            mask &= df["serious"].astype(bool)

        if self.hcp_only and "reporter_type" in df.columns:
            hcp_codes = {"MD", "HP", "PH", "OT"}
            mask &= df["reporter_type"].str.upper().isin(hcp_codes)

        if self.primary_suspect_only and "role_cod" in df.columns:
            mask &= df["role_cod"].str.upper() == "PS"

        return df[mask].copy()


# ---------------------------------------------------------------------------
# SignalResult
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    """Container for the output of a sigvigil analysis.

    Attributes
    ----------
    signals : DataFrame with one row per (drug, event) pair tested.
        Columns: drug, event, n_de, ror, ror_lo95, ror_hi95, ror_signal,
                 ic, ic025, ic_signal, prr, prr_chi2, prr_signal,
                 ebgm, eb05, ebgm_signal, p_value, bh_qvalue, storey_qvalue.
    population_filter : the Filter that was applied.
    drug_names : drugs analyzed.
    event_names : events analyzed.
    metadata : dict with run metadata (date, n_cases, FAERS quarters, etc.).
    """
    signals: pd.DataFrame
    population_filter: Filter
    drug_names: List[str]
    event_names: List[str]
    metadata: Dict = field(default_factory=dict)

    def summary(self, min_signals: int = 1) -> pd.DataFrame:
        """Return rows where at least min_signals methods flagged a signal."""
        sig_cols = ["ror_signal", "ic_signal", "prr_signal", "ebgm_signal"]
        present = [c for c in sig_cols if c in self.signals.columns]
        n_signals = self.signals[present].sum(axis=1)
        out = self.signals[n_signals >= min_signals].copy()
        out["n_methods_flagged"] = n_signals[n_signals >= min_signals]
        return out.sort_values("n_methods_flagged", ascending=False)

    def to_csv(self, path: str) -> None:
        self.signals.to_csv(path, index=False)
        logger.info("Signals written to %s", path)

    def to_json(self, path: str) -> None:
        self.signals.to_json(path, orient="records", indent=2)
        logger.info("Signals written to %s", path)

    def plot_heatmap(self, **kwargs):
        from sigvigil.viz.heatmap import plot_heatmap
        return plot_heatmap(self.signals, **kwargs)

    def plot_volcano(self, **kwargs):
        from sigvigil.viz.scatter import plot_volcano
        return plot_volcano(self.signals, **kwargs)

    def plot_timeline(self, db: "FAERSDatabase", **kwargs):
        from sigvigil.viz.timeline import plot_timeline
        return plot_timeline(db, self.drug_names, self.population_filter, **kwargs)

    def plot_ebgm_vs_ror(self, drug: str, **kwargs):
        from sigvigil.viz.scatter import plot_ebgm_vs_ror
        subset = self.signals[self.signals["drug"] == drug]
        return plot_ebgm_vs_ror(subset, drug=drug, **kwargs)

    def generate_report(self, output_path: str, **kwargs):
        from sigvigil.report.generator import generate_report
        generate_report(self, output_path, **kwargs)


# ---------------------------------------------------------------------------
# FAERSDatabase
# ---------------------------------------------------------------------------

class FAERSDatabase:
    """Primary interface to a preprocessed FAERS dataset.

    Load from a directory containing the output of FAERSPreprocessor,
    or from a single parquet/CSV case file.

    Parameters
    ----------
    cases : DataFrame with one row per deduplicated FAERS case.
        Required columns: caseid, drug_list (list of str), indication,
        reactions (list of str), age_years, sex, report_year.
    drugs_df : DataFrame with one row per drug-case link.
    reactions_df : DataFrame with one row per reaction-case link.
    metadata : dict with preprocessing provenance information.
    """

    def __init__(
        self,
        cases: pd.DataFrame,
        drugs_df: pd.DataFrame,
        reactions_df: pd.DataFrame,
        metadata: Optional[Dict] = None,
    ):
        self.cases = cases
        self.drugs_df = drugs_df
        self.reactions_df = reactions_df
        self.metadata = metadata or {}
        logger.info(
            "FAERSDatabase loaded: %d cases, %d drug-case links, %d reaction-case links",
            len(cases),
            len(drugs_df),
            len(reactions_df),
        )

    @classmethod
    def from_directory(cls, directory: str) -> "FAERSDatabase":
        """Load from a preprocessed sigvigil data directory."""
        p = Path(directory)
        cases_path = p / "cases.parquet"
        drugs_path = p / "drugs.parquet"
        reactions_path = p / "reactions.parquet"
        meta_path = p / "metadata.json"

        # Fallback to CSV if parquet not available
        if not cases_path.exists():
            cases_path = p / "cases.csv"
            drugs_path = p / "drugs.csv"
            reactions_path = p / "reactions.csv"

        if not cases_path.exists():
            raise FileNotFoundError(
                f"No cases file found in {directory}. "
                "Run FAERSPreprocessor first: see SETUP.md."
            )

        read = pd.read_parquet if str(cases_path).endswith(".parquet") else pd.read_csv
        cases = read(cases_path)
        drugs_df = read(drugs_path)
        reactions_df = read(reactions_path)

        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        return cls(cases, drugs_df, reactions_df, metadata)

    def apply_filter(self, filt: Optional[Filter]) -> "FAERSDatabase":
        """Return a new FAERSDatabase restricted to cases matching filt."""
        if filt is None:
            return self
        filtered_cases = filt.apply(self.cases)
        caseids = set(filtered_cases["caseid"])
        return FAERSDatabase(
            cases=filtered_cases,
            drugs_df=self.drugs_df[self.drugs_df["caseid"].isin(caseids)],
            reactions_df=self.reactions_df[
                self.reactions_df["caseid"].isin(caseids)
            ],
            metadata=self.metadata,
        )

    def analyze(
        self,
        drugs: Sequence[str],
        events: Sequence[str],
        population: Optional[Filter] = None,
        run_ebgm: bool = True,
        correction: str = "all",
    ) -> SignalResult:
        """Run full disproportionality analysis for given drug-event pairs.

        Parameters
        ----------
        drugs : list of canonical drug names (normalized).
        events : list of MedDRA preferred terms.
        population : Filter to restrict analysis population.
        run_ebgm : whether to fit and compute EBGM (slower).
        correction : 'bonferroni', 'bh', 'storey', or 'all'.

        Returns
        -------
        SignalResult with one row per (drug, event) pair.
        """
        import datetime

        target_db = self.apply_filter(population)
        N_total = len(target_db.cases)
        logger.info(
            "Running analysis on %d cases, %d drugs x %d events",
            N_total,
            len(drugs),
            len(events),
        )

        rows = []

        # Precompute per-drug and per-event counts in the filtered DB
        drug_case_counts = (
            target_db.drugs_df.groupby("drug_name_normalized")["caseid"]
            .nunique()
        )
        event_case_counts = (
            target_db.reactions_df.groupby("pt_name")["caseid"]
            .nunique()
        )

        for drug in drugs:
            for event in events:
                ct = build_contingency(
                    drug=drug,
                    event=event,
                    drugs_df=target_db.drugs_df,
                    reactions_df=target_db.reactions_df,
                    n_total_cases=N_total,
                )
                n_de = ct["n_de"]
                n_d = ct["n_d"]
                n_e = ct["n_e"]

                # --- ROR ---
                ror_res = compute_ror(
                    n_de=ct["n_de"],
                    n_d_not_e=ct["n_d_not_e"],
                    n_not_d_e=ct["n_not_d_e"],
                    n_not_d_not_e=ct["n_not_d_not_e"],
                )

                # --- IC ---
                ic_res = compute_ic(
                    n_de=n_de,
                    n_d=n_d,
                    n_e=n_e,
                    n_total=N_total,
                )

                # --- PRR ---
                prr_res = compute_prr(
                    n_de=ct["n_de"],
                    n_d=ct["n_d"],
                    n_not_d_e=ct["n_not_d_e"],
                    n_not_d=N_total - ct["n_d"],
                )

                row = {
                    "drug": drug,
                    "event": event,
                    "n_de": n_de,
                    "n_d": n_d,
                    "n_e": n_e,
                    "ror": ror_res["ror"],
                    "ror_lo95": ror_res["lo95"],
                    "ror_hi95": ror_res["hi95"],
                    "ror_signal": ror_res["signal"],
                    "ic": ic_res["ic"],
                    "ic025": ic_res["ic025"],
                    "ic_signal": ic_res["signal"],
                    "prr": prr_res["prr"],
                    "prr_chi2": prr_res["chi2"],
                    "prr_signal": prr_res["signal"],
                    "p_value": ror_res.get("p_value", np.nan),
                }
                rows.append(row)

        results = pd.DataFrame(rows)

        # --- EBGM (fit once across all pairs, compute per pair) ---
        if run_ebgm and len(results) > 0:
            try:
                all_cells = []
                for drug in drugs:
                    all_cells.extend(
                        self._build_all_cells_for_drug(drug, target_db, N_total)
                    )
                if len(all_cells) >= 50:
                    prior_params = fit_ebgm_prior(all_cells)
                    ebgm_vals = []
                    eb05_vals = []
                    for _, row in results.iterrows():
                        ct = build_contingency(
                            drug=row["drug"],
                            event=row["event"],
                            drugs_df=target_db.drugs_df,
                            reactions_df=target_db.reactions_df,
                            n_total_cases=N_total,
                        )
                        n_de = ct["n_de"]
                        e_de = (ct["n_d"] * ct["n_e"]) / max(N_total, 1)
                        eb = compute_ebgm_posterior(n_de, e_de, prior_params)
                        ebgm_vals.append(eb["ebgm"])
                        eb05_vals.append(eb["eb05"])
                    results["ebgm"] = ebgm_vals
                    results["eb05"] = eb05_vals
                    results["ebgm_signal"] = results["eb05"] >= 2.0
                else:
                    warnings.warn(
                        "Too few drug-event pairs to fit EBGM prior reliably. "
                        "EBGM skipped."
                    )
                    results["ebgm"] = np.nan
                    results["eb05"] = np.nan
                    results["ebgm_signal"] = False
            except Exception as e:
                warnings.warn(f"EBGM computation failed: {e}. Skipping.")
                results["ebgm"] = np.nan
                results["eb05"] = np.nan
                results["ebgm_signal"] = False

        # --- Multiple testing correction ---
        if "p_value" in results.columns:
            pvals = results["p_value"].fillna(1.0).values
            if correction in ("bonferroni", "all"):
                results["bonferroni_significant"] = bonferroni(pvals)
            if correction in ("bh", "all"):
                results["bh_qvalue"] = benjamini_hochberg(pvals)
                results["bh_significant"] = results.get("bh_qvalue", pd.Series(np.ones(len(results)))) <= 0.05
            if correction in ("storey", "all"):
                try:
                    results["storey_qvalue"] = storey_qvalue(pvals)
                    results["storey_significant"] = results["storey_qvalue"] <= 0.05
                except Exception:
                    results["storey_qvalue"] = np.nan

        meta = {
            "run_date": str(pd.Timestamp.now().date()),
            "n_cases_analyzed": N_total,
            "n_drug_event_pairs": len(results),
            "drugs_analyzed": list(drugs),
            "events_analyzed": list(events),
            "faers_metadata": self.metadata,
        }

        return SignalResult(
            signals=results,
            population_filter=population or Filter(),
            drug_names=list(drugs),
            event_names=list(events),
            metadata=meta,
        )

    def _build_all_cells_for_drug(
        self, drug: str, db: "FAERSDatabase", n_total: int
    ) -> List[Dict]:
        """Build (n_de, e_de) for all events co-occurring with drug."""
        drug_cases = set(
            db.drugs_df[
                db.drugs_df["drug_name_normalized"] == drug
            ]["caseid"]
        )
        n_d = len(drug_cases)

        all_events = db.reactions_df["pt_name"].unique()
        cells = []
        for evt in all_events:
            evt_cases = set(
                db.reactions_df[db.reactions_df["pt_name"] == evt]["caseid"]
            )
            n_e = len(evt_cases)
            n_de = len(drug_cases & evt_cases)
            e_de = (n_d * n_e) / max(n_total, 1)
            cells.append({"n_de": n_de, "e_de": e_de})
        return cells

    def reporting_rate_by_year(
        self, drugs: Sequence[str], population: Optional[Filter] = None
    ) -> pd.DataFrame:
        """Return report counts per year per drug for timeline visualization."""
        db = self.apply_filter(population)
        records = []
        for drug in drugs:
            drug_cases = db.drugs_df[
                db.drugs_df["drug_name_normalized"] == drug
            ][["caseid"]].merge(
                db.cases[["caseid", "report_year"]], on="caseid", how="left"
            )
            yearly = drug_cases.groupby("report_year").size().reset_index(name="n_reports")
            yearly["drug"] = drug
            records.append(yearly)
        if not records:
            return pd.DataFrame(columns=["report_year", "n_reports", "drug"])
        return pd.concat(records, ignore_index=True)
