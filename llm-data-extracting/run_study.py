from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliability.analysis import run_failure_analysis
from reliability.benchmark import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run invoice extraction reliability benchmark end-to-end."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "dataset" / "batch_1" / "batch_1",
        help="Path to CSV dataset directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results",
        help="Path where results are written.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Repeated runs per setup.")
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.25,
        help="Fraction of documents used for evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting/perturbations.")
    parser.add_argument(
        "--no-stress-test",
        action="store_true",
        help="Disable OCR perturbation stress variants.",
    )
    parser.add_argument(
        "--use-production-gpt4o",
        action="store_true",
        help="Use OpenAI GPT-4o extractor instead of local simulator.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Limit number of eval docs (0 = all). Useful for rate-limit-safe smoke tests.",
    )
    parser.add_argument(
        "--fail-on-api-error",
        action="store_true",
        help="Fail the run if GPT API calls keep failing after retries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        runs=args.runs,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        run_stress_test=not args.no_stress_test,
        use_production_gpt4o=args.use_production_gpt4o,
        max_docs=args.max_docs,
        fail_on_api_error=args.fail_on_api_error,
    )
    run_failure_analysis(dataset_root=args.dataset_root, output_root=args.output_root)
    print(f"Wrote benchmark + analysis artifacts to: {args.output_root}")


if __name__ == "__main__":
    main()

