"""RECS workflow runner — execute all analysis steps in sequence.

Runs steps 01 through 06 in order.  Step 06 (cross-dataset comparison)
requires at least one external dataset (--fm-data or --rbsa-dir) and is
skipped automatically if neither is provided.

Usage::

    # Minimal run (steps 01–05 only, multifamily units)
    python analysis/recs/run_workflow.py \\
        --data path/to/recs2020_public_v7.csv \\
        --outdir outputs/recs

    # Full run with FM and RBSA cross-dataset comparison
    python analysis/recs/run_workflow.py \\
        --data path/to/recs2020_public_v7.csv \\
        --outdir outputs/recs \\
        --fm-data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \\
        --rbsa-dir path/to/2022_RBSA_Datasets/

    # Single-family analysis
    python analysis/recs/run_workflow.py \\
        --data path/to/recs2020_public_v7.csv \\
        --outdir outputs/recs \\
        --unit-type sf

    # Skip step 01 if curated parquet already exists
    python analysis/recs/run_workflow.py \\
        --data path/to/recs2020_public_v7.csv \\
        --outdir outputs/recs \\
        --skip-steps 01
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

# Resolve the repo root so child scripts can import src.*
_REPO_ROOT = Path(__file__).resolve().parents[2]

STEPS = ["01", "02", "03", "04", "05", "06"]

SCRIPT_MAP = {
    "01": _REPO_ROOT / "analysis" / "recs" / "01_build_curated_table.py",
    "02": _REPO_ROOT / "analysis" / "recs" / "02_exploratory_distributions.py",
    "03": _REPO_ROOT / "analysis" / "recs" / "03_statistical_testing.py",
    "04": _REPO_ROOT / "analysis" / "recs" / "04_climate_vintage_controlled.py",
    "05": _REPO_ROOT / "analysis" / "recs" / "05_fuel_breakdown.py",
    "06": _REPO_ROOT / "analysis" / "recs" / "06_compare_with_other.py",
}


def _find_curated(outdir: Path) -> str | None:
    """Return the path to the most recent curated RECS parquet, or None."""
    matches = sorted(glob.glob(str(outdir / "recs2020_curated_*.parquet")))
    return matches[-1] if matches else None


def _run(step: str, cmd: list[str], dry_run: bool) -> None:
    label = f"[step {step}]"
    print(f"\n{'=' * 60}")
    print(f"  {label} {' '.join(str(c) for c in cmd)}")
    print("=" * 60)
    if dry_run:
        print("  (dry-run — skipping execution)")
        return
    result = subprocess.run(cmd, cwd=_REPO_ROOT)
    if result.returncode != 0:
        print(f"\n  ERROR: step {step} exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the full RECS 2020 analysis workflow (steps 01–06).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--data", type=Path, required=True, metavar="CSV",
                    help="Path to recs2020_public_v7.csv (or similar public-use file).")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/recs"),
                    help="Output directory (default: outputs/recs).")
    ap.add_argument("--unit-type", choices=["mf", "sf", "all"], default="mf",
                    help="Housing unit filter for steps 01–06 (default: mf).")

    # Optional cross-dataset comparison inputs (step 06)
    ap.add_argument("--fm-data", type=Path, default=None,
                    help="Path to FM survey Excel file (optional, for step 06).")
    rbsa_src = ap.add_mutually_exclusive_group()
    rbsa_src.add_argument("--rbsa-dir", type=Path, default=None,
                          help="Directory with extracted RBSA CSVs (optional, for step 06).")
    rbsa_src.add_argument("--rbsa-zip", type=Path, default=None,
                          help="Path to RBSA zip archive (optional, for step 06).")

    ap.add_argument("--skip-steps", nargs="*", metavar="STEP", default=[],
                    choices=STEPS,
                    help="Steps to skip, e.g. --skip-steps 01 06.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing them.")

    args = ap.parse_args()
    skip = set(args.skip_steps or [])

    # Auto-skip step 06 if no comparison datasets provided
    has_comparison = args.fm_data is not None or args.rbsa_dir is not None or args.rbsa_zip is not None
    if not has_comparison and "06" not in skip:
        print("  Note: no comparison datasets provided (--fm-data / --rbsa-dir / --rbsa-zip) "
              "— skipping step 06.")
        skip.add("06")

    py = sys.executable

    for step in STEPS:
        if step in skip:
            print(f"\n  Skipping step {step}.")
            continue

        script = str(SCRIPT_MAP[step])

        if step == "01":
            cmd = [
                py, script,
                "--data", str(args.data),
                "--outdir", str(args.outdir),
                "--unit-type", args.unit_type,
            ]

        elif step in ("02", "03", "04", "05"):
            curated = _find_curated(args.outdir)
            if curated is None:
                print(f"\n  ERROR: curated parquet not found in {args.outdir}. Run step 01 first.")
                sys.exit(1)
            cmd = [
                py, script,
                "--curated", curated,
                "--outdir", str(args.outdir),
                "--unit-type", args.unit_type,
            ]

        elif step == "06":
            curated = _find_curated(args.outdir)
            if curated is None:
                print(f"\n  ERROR: curated parquet not found in {args.outdir}. Run step 01 first.")
                sys.exit(1)
            cmd = [
                py, script,
                "--recs-curated", curated,
                "--outdir", str(args.outdir),
            ]
            if args.fm_data:
                cmd += ["--fm-data", str(args.fm_data)]
            if args.rbsa_dir:
                cmd += ["--rbsa-dir", str(args.rbsa_dir)]
            elif args.rbsa_zip:
                cmd += ["--rbsa-zip", str(args.rbsa_zip)]

        _run(step, cmd, args.dry_run)

    print("\n  All steps complete.")


if __name__ == "__main__":
    main()
