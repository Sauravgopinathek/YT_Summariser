"""Human Evaluation Framework and Metrics Aggregator.

Computes empirical qualitative assessment metrics across three key dimensions:
1. Completeness (1-5): Preserves important formulas, steps, concepts, and terminology.
2. Scannability (1-5): Clear structure using Executive Summary, Key Concepts, and Action Items.
3. Fact Verification (1-5): Claims can be verified directly against the original transcript.

Calculates overall averages and per-video averages without fabricating evaluator responses.
"""

import argparse
import csv
import logging
from pathlib import Path
import statistics
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "human_evaluation_template.csv"
DEFAULT_EVALUATION_PATH = Path(__file__).resolve().parent / "human_evaluations.csv"

EVALUATION_COLUMNS = [
    "video_id",
    "evaluator_id",
    "completeness_score",
    "scannability_score",
    "fact_verification_score",
    "comments",
]

SCORE_FIELDS = [
    "completeness_score",
    "scannability_score",
    "fact_verification_score",
]


def load_human_evaluations(csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load and validate human evaluation rows from a CSV file.

    Skips empty/unrated template rows where scores have not yet been entered.

    Args:
        csv_path: Path to the evaluation CSV.

    Returns:
        List of parsed evaluation dictionaries with float scores.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If scores are outside the valid 1-5 scale.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    valid_evaluations: List[Dict[str, Any]] = []

    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            video_id = row.get("video_id", "").strip()
            evaluator_id = row.get("evaluator_id", "").strip()

            comp_raw = row.get("completeness_score", "").strip()
            scan_raw = row.get("scannability_score", "").strip()
            fact_raw = row.get("fact_verification_score", "").strip()
            comments = row.get("comments", "").strip()

            # Skip unrated rows (e.g. fresh template rows)
            if not comp_raw and not scan_raw and not fact_raw:
                continue

            # Parse scores
            scores: Dict[str, float] = {}
            for field, raw_val in [
                ("completeness", comp_raw),
                ("scannability", scan_raw),
                ("fact_verification", fact_raw),
            ]:
                if not raw_val:
                    raise ValueError(
                        f"Row {row_idx} (video: {video_id}): Missing score for '{field}'."
                    )
                try:
                    val = float(raw_val)
                except ValueError:
                    raise ValueError(
                        f"Row {row_idx} (video: {video_id}): Non-numeric score '{raw_val}' for '{field}'."
                    )

                if val < 1.0 or val > 5.0:
                    raise ValueError(
                        f"Row {row_idx} (video: {video_id}): Score {val} for '{field}' is out of range [1, 5]."
                    )
                scores[field] = val

            valid_evaluations.append(
                {
                    "video_id": video_id,
                    "evaluator_id": evaluator_id or "anonymous",
                    "completeness_score": scores["completeness"],
                    "scannability_score": scores["scannability"],
                    "fact_verification_score": scores["fact_verification"],
                    "comments": comments,
                }
            )

    return valid_evaluations


def calculate_human_eval_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall and per-video averages from human evaluation records.

    Args:
        records: List of validated evaluation dictionaries.

    Returns:
        Dictionary containing overall and per-video metrics.
    """
    if not records:
        return {
            "total_evaluations": 0,
            "unique_videos": 0,
            "unique_evaluators": 0,
            "average_completeness": 0.0,
            "average_scannability": 0.0,
            "average_fact_verification": 0.0,
            "overall_average": 0.0,
            "per_video": {},
        }

    total_evals = len(records)
    unique_videos = sorted(list({r["video_id"] for r in records if r["video_id"]}))
    unique_evaluators = sorted(list({r["evaluator_id"] for r in records if r["evaluator_id"]}))

    all_completeness = [r["completeness_score"] for r in records]
    all_scannability = [r["scannability_score"] for r in records]
    all_fact_verif = [r["fact_verification_score"] for r in records]

    avg_comp = round(statistics.mean(all_completeness), 2)
    avg_scan = round(statistics.mean(all_scannability), 2)
    avg_fact = round(statistics.mean(all_fact_verif), 2)
    overall_avg = round((avg_comp + avg_scan + avg_fact) / 3.0, 2)

    # Per-video metrics breakdown
    per_video: Dict[str, Dict[str, Any]] = {}
    for vid in unique_videos:
        vid_records = [r for r in records if r["video_id"] == vid]
        v_comp = [r["completeness_score"] for r in vid_records]
        v_scan = [r["scannability_score"] for r in vid_records]
        v_fact = [r["fact_verification_score"] for r in vid_records]

        v_avg_comp = round(statistics.mean(v_comp), 2)
        v_avg_scan = round(statistics.mean(v_scan), 2)
        v_avg_fact = round(statistics.mean(v_fact), 2)
        v_overall = round((v_avg_comp + v_avg_scan + v_avg_fact) / 3.0, 2)

        per_video[vid] = {
            "evaluation_count": len(vid_records),
            "average_completeness": v_avg_comp,
            "average_scannability": v_avg_scan,
            "average_fact_verification": v_avg_fact,
            "overall_average": v_overall,
        }

    return {
        "total_evaluations": total_evals,
        "unique_videos": len(unique_videos),
        "unique_evaluators": len(unique_evaluators),
        "average_completeness": avg_comp,
        "average_scannability": avg_scan,
        "average_fact_verification": avg_fact,
        "overall_average": overall_avg,
        "per_video": per_video,
    }


def format_human_eval_report(metrics: Dict[str, Any]) -> str:
    """Format human evaluation metrics into a publication-ready Markdown report."""
    if metrics["total_evaluations"] == 0:
        return (
            "# Human Evaluation Report\n\n"
            "**No completed evaluations found.**\n\n"
            "To record ratings, fill in scores (1-5 scale) in `evaluation/human_evaluation_template.csv`."
        )

    lines = [
        "# Human Evaluation Report (1-5 Rubric Scale)",
        "",
        "### Evaluation Dimensions",
        "- **Completeness (1-5)**: Preservation of formulas, mechanisms, concepts, and technical terminology.",
        "- **Scannability (1-5)**: Visual clarity across Executive Summary, Key Concepts, and Action Items.",
        "- **Fact Verification (1-5)**: Grounded claims directly verifiable against source transcripts.",
        "",
        "### Overall Summary",
        "",
        "| Metric | Score / Count |",
        "| :--- | :--- |",
        f"| **Total Evaluations** | {metrics['total_evaluations']} |",
        f"| **Evaluated Videos** | {metrics['unique_videos']} |",
        f"| **Unique Evaluators** | {metrics['unique_evaluators']} |",
        f"| **Average Completeness** | **{metrics['average_completeness']:.2f}** / 5.0 |",
        f"| **Average Scannability** | **{metrics['average_scannability']:.2f}** / 5.0 |",
        f"| **Average Fact Verification** | **{metrics['average_fact_verification']:.2f}** / 5.0 |",
        f"| **Overall Composite Score** | **{metrics['overall_average']:.2f}** / 5.0 |",
        "",
        "### Per-Video Breakdown",
        "",
        "| Video ID | Evaluations | Completeness (1-5) | Scannability (1-5) | Fact Verification (1-5) | Overall |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for vid, stats in metrics["per_video"].items():
        lines.append(
            f"| `{vid}` | {stats['evaluation_count']} | "
            f"{stats['average_completeness']:.2f} | "
            f"{stats['average_scannability']:.2f} | "
            f"{stats['average_fact_verification']:.2f} | "
            f"**{stats['overall_average']:.2f}** |"
        )

    return "\n".join(lines)


def main():
    """CLI entrypoint for computing human evaluation metrics."""
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Calculate Human Evaluation Metrics from Evaluation CSV."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to human evaluation CSV file. Defaults to template or evaluations file.",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default=None,
        help="Optional path to save Markdown report.",
    )

    args = parser.parse_args()

    # Determine input file
    input_file = Path(args.file) if args.file else (
        DEFAULT_EVALUATION_PATH if DEFAULT_EVALUATION_PATH.exists() else DEFAULT_TEMPLATE_PATH
    )

    print(f"Reading human evaluations from: {input_file}")
    records = load_human_evaluations(input_file)
    metrics = calculate_human_eval_metrics(records)
    report_md = format_human_eval_report(metrics)

    print("\n" + report_md)


    if args.out:
        out_path = Path(args.out)
        out_path.write_text(report_md, encoding="utf-8")
        print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
