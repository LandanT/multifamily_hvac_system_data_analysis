"""RBSA workflow runner — execute all analysis steps in sequence.

Runs steps 00 through 06 in order.  Step 00 (sanity checks) and step 06
(cross-dataset comparison) are optional and can be skipped.

Usage::

    # Minimal run (steps 01–05 only, MF buildings)
    python analysis/rbsa/run_workflow.py \\
        --data-dir path/to/2022_RBSA_Datasets/ \\
        --outdir outputs/rbsa

    # Full run including FM cross-dataset comparison
    python analysis/rbsa/run_workflow.py \\
        --data-dir path/to/2022_RBSA_Datasets/ \\
        --outdir outputs/rbsa \\
        --fm-data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx

    # Run from a zip file, skip sanity checks
    python analysis/rbsa/run_workflow.py \\
        --zip path/to/rbsa_2022.zip \\
        --outdir outputs/rbsa \\
        --skip-steps 00

    # Single-family analysis
    python analysis/rbsa/run_workflow.py \\
        --data-dir path/to/2022_RBSA_Datasets/ \\
        --outdir outputs/rbsa \\
        --building-type sf
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

# Resolve the repo root so child scripts can import src.*
_REPO_ROOT = Path(__file__).resolve().parents[2]

STEPS = ["00", "01", "02", "03", "04", "05", "06"]

SCRIPT_MAP = {
    "00": _REPO_ROOT / "analysis" / "rbsa" / "00_sanity_checks.py",
    "01": _REPO_ROOT / "analysis" / "rbsa" / "01_build_curated_mf_table.py",
    "02": _REPO_ROOT / "analysis" / "rbsa" / "02_exploratory_distributions.py",
    "03": _REPO_ROOT / "analysis" / "rbsa" / "03_statistical_testing.py",
    "04": _REPO_ROOT / "analysis" / "rbsa" / "04_climate_vintage_controlled.py",
    "05": _REPO_ROOT / "analysis" / "rbsa" / "05_fuel_breakdown.py",
    "06": _REPO_ROOT / "analysis" / "rbsa" / "06_compare_with_fm.py",
}


def _find_site_master(outdir: Path) -> str | None:
    """Return the path to the most recent site master parquet, or None."""
    matches = sorted(glob.glob(str(outdir / "rbsa_site_master_*.parquet")))
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
        description="Run the full RBSA analysis workflow (steps 00–06).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source (mutually exclusive: zip or directory)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", type=Path, metavar="DIR",
                     help="Directory containing extracted 2022 RBSA dataset CSVs.")
    src.add_argument("--zip", type=Path, metavar="ZIP",
                     help="Path to RBSA zip archive.")

    ap.add_argument("--outdir", type=Path, default=Path("outputs/rbsa"),
                    help="Output directory (default: outputs/rbsa).")
    ap.add_argument("--building-type", choices=["mf", "sf", "all"], default="mf",
                    help="Building-type filter for steps 02–06 (default: mf).")
    ap.add_argument("--fm-data", type=Path, default=None,
                    help="Path to FM survey Excel file (required for step 06).")
    ap.add_argument("--skip-steps", nargs="*", metavar="STEP", default=[],
                    choices=STEPS,
                    help="Steps to skip, e.g. --skip-steps 00 06.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing them.")

    args = ap.parse_args()
    skip = set(args.skip_steps or [])

    # If no FM data, auto-skip step 06
    if args.fm_data is None and "06" not in skip:
        print("  Note: --fm-data not provided — skipping step 06 (cross-dataset comparison).")
        skip.add("06")

    py = sys.executable

    for step in STEPS:
        if step in skip:
            print(f"\n  Skipping step {step}.")
            continue

        script = str(SCRIPT_MAP[step])

        if step == "00":
            site_master = _find_site_master(args.outdir)
            if site_master is None:
                print(f"\n  Step 00: no site master found in {args.outdir} — skipping sanity check.")
                continue
            cmd = [py, script, "--curated", site_master, "--table-type", "site_master"]

        elif step == "01":
            cmd = [py, script, "--outdir", str(args.outdir)]
            if args.data_dir:
                cmd += ["--data-dir", str(args.data_dir)]
            else:
                cmd += ["--zip", str(args.zip)]

        elif step in ("02", "03", "04", "05"):
            site_master = _find_site_master(args.outdir)
            if site_master is None:
                print(f"\n  ERROR: site master parquet not found in {args.outdir}. Run step 01 first.")
                sys.exit(1)
            cmd = [
                py, script,
                "--site-master", site_master,
                "--outdir", str(args.outdir),
                "--building-type", args.building_type,
            ]

        elif step == "06":
            site_master = _find_site_master(args.outdir)
            if site_master is None:
                print(f"\n  ERROR: site master parquet not found in {args.outdir}. Run step 01 first.")
                sys.exit(1)
            cmd = [
                py, script,
                "--site-master", site_master,
                "--fm-data", str(args.fm_data),
                "--outdir", str(args.outdir),
                "--building-type", args.building_type,
            ]

        _run(step, cmd, args.dry_run)

    print("\n  All steps complete.")


if __name__ == "__main__":
    main()
