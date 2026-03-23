"""
Generate the full analysis report from experiment results.

Reads all CSVs from results/, produces figures and a summary report.

Usage:
    python -m scripts.generate_report
    python -m scripts.generate_report --results-dir results --figures-dir results/figures
"""

import argparse

from evaluation.analyzer import Analyzer


def main():
    parser = argparse.ArgumentParser(description="Generate full Azul analysis report")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing experiment CSV files")
    parser.add_argument("--figures-dir", type=str, default=None,
                        help="Directory for output figures (default: <results-dir>/figures)")
    args = parser.parse_args()

    analyzer = Analyzer(results_dir=args.results_dir, figures_dir=args.figures_dir)
    report_path = analyzer.generate_full_report()
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
