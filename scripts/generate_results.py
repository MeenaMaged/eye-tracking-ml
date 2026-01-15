#!/usr/bin/env python3
"""
Generate all results for Eye Tracking ML Project documentation.

This script:
1. Runs all model evaluations
2. Generates comparison tables (Markdown + CSV)
3. Creates all plots (confusion matrices, ROC curves, etc.)
4. Compiles everything into FINAL_REPORT_DATA.md

Run: python scripts/generate_results.py [--synthetic]
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def main():
    """Main entry point for results generation."""
    parser = argparse.ArgumentParser(
        description="Generate all results for Eye Tracking ML Project"
    )
    parser.add_argument(
        "--synthetic", "-s",
        action="store_true",
        help="Use synthetic data if real datasets not available"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data",
        help="Directory containing datasets (default: data)"
    )
    parser.add_argument(
        "--results-dir", "-r",
        default="results",
        help="Directory for output results (default: results)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation (faster, text-only results)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("EYE TRACKING ML PROJECT - RESULTS GENERATOR")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Use synthetic data: {args.synthetic}")
    print("=" * 70)

    # Import and run batch evaluation
    from integration.batch_evaluation import run_full_evaluation

    evaluation = run_full_evaluation(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_synthetic=args.synthetic,
        verbose=not args.quiet
    )

    print("\n" + "=" * 70)
    print("RESULTS GENERATION COMPLETE")
    print("=" * 70)

    # Print output locations
    print("\nGenerated files:")
    print(f"  {args.results_dir}/")
    print(f"    - FINAL_REPORT_DATA.md        (Combined results)")
    print(f"    - all_results.json            (JSON format)")
    print(f"    - phase3_gaze/")
    print(f"        - regression_comparison.md")
    print(f"        - regression_comparison.csv")
    print(f"    - phase4_blink/")
    print(f"        - classifier_comparison.md")
    print(f"        - plots/")
    print(f"    - phase5_movement/")
    print(f"        - classifier_comparison.md")
    print(f"        - plots/")
    print(f"    - phase6_features/")
    print(f"        - pca_analysis.md")
    print(f"        - plots/")
    print(f"    - phase7_evaluation/")
    print(f"        - cross_validation.md")

    print("\n" + "=" * 70)
    print("Copy these files to your report:")
    print("  - Tables: results/*/*.md")
    print("  - Figures: results/*/plots/*.png")
    print("  - Data: results/*/*.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
