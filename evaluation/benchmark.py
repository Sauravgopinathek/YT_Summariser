"""Benchmark execution and telemetry reporting module.

Provides a comprehensive 25-video benchmark framework that processes video datasets
through the production zero-download pipeline, records stage-by-stage latencies,
compression ratios, ROUGE scores (only when ground-truth references exist),
and generates structured empirical telemetry reports in CSV and Markdown formats.
"""

import argparse
import csv
import logging
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

from evaluation.metrics import (
    PerformanceMetrics,
    calculate_compression_ratio,
    calculate_rouge,
)
from modules.chunking import chunk_transcript
from modules.formatter import format_as_markdown, validate_markdown_notes
from modules.llm import NVIDIAClient
from modules.preprocessing import clean_transcript
from modules.summarizer import generate_notes
from modules.transcript import TranscriptProvider, YouTubeTranscriptProvider
from modules.url_parser import extract_video_id
from modules.validation import validate_summary_faithfulness
from utils.error_handler import sanitize_secrets

logger = logging.getLogger(__name__)

# Standard File Locations
DEFAULT_DATASET_CSV = Path(__file__).resolve().parent / "test_dataset.csv"
DEFAULT_RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"
BENCHMARK_CSV_PATH = Path(__file__).resolve().parent / "benchmark_results.csv"

# Legacy benchmark row headers
CSV_HEADERS = [
    "video_id",
    "category",
    "transcript_words",
    "summary_words",
    "compression_ratio",
    "chunk_count",
    "llm_calls",
    "latency",
    "status",
]

# Results CSV headers for the 25-video dataset evaluation
RESULTS_CSV_HEADERS = [
    "video_id",
    "youtube_url",
    "domain",
    "approx_duration",
    "status",
    "transcript_words",
    "summary_words",
    "compression_ratio",
    "chunk_count",
    "llm_calls",
    "total_latency_seconds",
    "retrieval_latency",
    "preprocessing_latency",
    "chunking_latency",
    "llm_latency",
    "rouge1_f1",
    "rouge2_f1",
    "rougel_f1",
    "error",
    "notes",
]


class BenchmarkRunner:
    """Automated benchmark executor for the YouTube video summarization pipeline."""

    def __init__(
        self,
        output_csv_path: Optional[Path] = None,
        llm_client: Optional[NVIDIAClient] = None,
        transcript_provider: Optional[TranscriptProvider] = None,
        results_csv_path: Optional[Path] = None,
    ):
        self.output_csv = output_csv_path or BENCHMARK_CSV_PATH
        self.results_csv = results_csv_path or DEFAULT_RESULTS_CSV
        self.llm_client = llm_client or NVIDIAClient()
        self.transcript_provider = transcript_provider or YouTubeTranscriptProvider()
        self._ensure_csv_header(self.output_csv, CSV_HEADERS)
        self._ensure_csv_header(self.results_csv, RESULTS_CSV_HEADERS)

    @staticmethod
    def _ensure_csv_header(csv_path: Path, headers: List[str]) -> None:
        """Initialize a CSV file with headers if it does not exist."""
        if not csv_path.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def append_benchmark_row(
        self,
        video_id: str,
        category: str,
        transcript_words: int,
        summary_words: int,
        compression_ratio: float,
        chunk_count: int,
        llm_calls: int,
        latency: float,
        status: str,
    ) -> None:
        """Append an empirical observation row to benchmark_results.csv."""
        with open(self.output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    video_id,
                    category or "general",
                    transcript_words,
                    summary_words,
                    f"{compression_ratio:.4f}",
                    chunk_count,
                    llm_calls,
                    f"{latency:.2f}",
                    status,
                ]
            )

    def append_results_row(self, row_data: Dict[str, Any]) -> None:
        """Append a processed video record to results.csv without fabricating values."""
        with open(self.results_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    row_data.get("video_id", ""),
                    row_data.get("youtube_url", ""),
                    row_data.get("domain", ""),
                    row_data.get("approx_duration", ""),
                    row_data.get("status", ""),
                    row_data.get("transcript_words", ""),
                    row_data.get("summary_words", ""),
                    (
                        f"{row_data['compression_ratio']:.4f}"
                        if row_data.get("compression_ratio") is not None
                        else ""
                    ),
                    row_data.get("chunk_count", ""),
                    row_data.get("llm_calls", ""),
                    (
                        f"{row_data['total_latency_seconds']:.2f}"
                        if row_data.get("total_latency_seconds") is not None
                        else ""
                    ),
                    (
                        f"{row_data['retrieval_latency']:.4f}"
                        if row_data.get("retrieval_latency") is not None
                        else ""
                    ),
                    (
                        f"{row_data['preprocessing_latency']:.4f}"
                        if row_data.get("preprocessing_latency") is not None
                        else ""
                    ),
                    (
                        f"{row_data['chunking_latency']:.4f}"
                        if row_data.get("chunking_latency") is not None
                        else ""
                    ),
                    (
                        f"{row_data['llm_latency']:.4f}"
                        if row_data.get("llm_latency") is not None
                        else ""
                    ),
                    (
                        f"{row_data['rouge1_f1']:.4f}"
                        if row_data.get("rouge1_f1") is not None
                        else ""
                    ),
                    (
                        f"{row_data['rouge2_f1']:.4f}"
                        if row_data.get("rouge2_f1") is not None
                        else ""
                    ),
                    (
                        f"{row_data['rougel_f1']:.4f}"
                        if row_data.get("rougel_f1") is not None
                        else ""
                    ),
                    row_data.get("error", ""),
                    row_data.get("notes", ""),
                ]
            )

    def run_benchmark(
        self,
        youtube_url: str,
        category: str = "general",
        reference_summary: Optional[str] = None,
        max_words_per_chunk: Optional[int] = None,
        overlap_words: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a full performance and quality benchmark run on a single video.

        Measures latency at every pipeline step, computes word counts and compression,
        and optionally evaluates ROUGE if a genuine ground-truth reference is provided.
        """
        metrics = PerformanceMetrics()
        e2e_start = time.perf_counter()
        video_id = "unknown"
        status = "FAILED"
        formatted_notes = None
        rouge_scores = None

        try:
            # 1. URL Parsing Latency
            t0 = time.perf_counter()
            video_id = extract_video_id(youtube_url)
            metrics.url_parsing_latency_seconds = round(time.perf_counter() - t0, 4)

            # 2. Transcript Retrieval Latency
            t0 = time.perf_counter()
            raw_segments = self.transcript_provider.get_transcript(video_id)
            metrics.transcript_retrieval_latency_seconds = round(time.perf_counter() - t0, 4)

            # 3. Preprocessing Latency
            t0 = time.perf_counter()
            preprocessed = clean_transcript(raw_segments)
            clean_text = preprocessed["clean_text"]
            metrics.preprocessing_latency_seconds = round(time.perf_counter() - t0, 4)
            metrics.input_word_count = preprocessed["word_count"]

            # 4. Chunking Latency
            t0 = time.perf_counter()
            chunks = chunk_transcript(
                clean_text,
                max_words=max_words_per_chunk,
                overlap_words=overlap_words,
            )
            metrics.chunking_latency_seconds = round(time.perf_counter() - t0, 4)
            metrics.chunk_count = len(chunks)

            # 5. LLM Inference Latency
            t0 = time.perf_counter()
            notes_result = generate_notes(
                transcript=clean_text,
                client=self.llm_client,
                max_words_per_chunk=max_words_per_chunk,
                overlap_words=overlap_words,
            )
            metrics.llm_inference_latency_seconds = round(time.perf_counter() - t0, 4)

            # 6. Formatting & Output Validation
            formatted_notes = format_as_markdown(notes_result)
            validate_markdown_notes(formatted_notes)

            # Metrics Computation
            metrics.llm_call_count = notes_result["metadata"]["llm_call_count"]
            metrics.output_word_count = len(formatted_notes.split())

            ratio, reduction = calculate_compression_ratio(
                metrics.input_word_count,
                metrics.output_word_count,
            )
            metrics.compression_ratio = ratio
            metrics.reduction_percentage = reduction

            # Grounding Audit Score
            audit = validate_summary_faithfulness(clean_text, formatted_notes)
            metrics.faithfulness_score = audit["faithfulness_score"]

            # Optional ROUGE Evaluation (Only when genuine reference is supplied)
            if reference_summary and reference_summary.strip():
                rouge_scores = calculate_rouge(formatted_notes, reference_summary.strip())

            status = "SUCCESS"

        except Exception as e:
            logger.error("Benchmark failed for video %s: %s", video_id, e)
            status = f"FAILED: {type(e).__name__}"
            raise

        finally:
            total_elapsed = round(time.perf_counter() - e2e_start, 2)
            metrics.total_processing_time_seconds = total_elapsed

            self.append_benchmark_row(
                video_id=video_id,
                category=category,
                transcript_words=metrics.input_word_count,
                summary_words=metrics.output_word_count,
                compression_ratio=metrics.compression_ratio,
                chunk_count=metrics.chunk_count,
                llm_calls=metrics.llm_call_count,
                latency=total_elapsed,
                status=status,
            )

        return {
            "video_id": video_id,
            "category": category,
            "status": status,
            "metrics": metrics.to_dict(),
            "rouge": rouge_scores,
            "notes": formatted_notes if status == "SUCCESS" else None,
        }

    def process_dataset_record(
        self,
        record: Dict[str, str],
        max_words_per_chunk: Optional[int] = None,
        overlap_words: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process a single record from test_dataset.csv through the 12 pipeline steps.

        Steps executed:
        1. Validate URL.
        2. Extract video ID.
        3. Retrieve transcript.
        4. Preprocess transcript.
        5. Chunk transcript.
        6. Generate summary.
        7. Record latency (overall & per-stage).
        8. Record transcript size.
        9. Record summary size.
        10. Calculate compression ratio.
        11. Record success/failure.
        12. Calculate ROUGE only when reference summaries are available.

        Returns:
            Dictionary matching the results.csv schema and evaluation metrics.
        """
        youtube_url = record.get("youtube_url", "").strip()
        expected_video_id = record.get("video_id", "").strip()
        domain = record.get("domain", "").strip()
        approx_duration = record.get("approx_duration", "").strip()
        reference_summary = record.get("reference_summary", "").strip()
        notes_info = record.get("notes", "").strip()

        metrics = PerformanceMetrics()
        e2e_start = time.perf_counter()
        extracted_id = expected_video_id or "unknown"
        status = "FAILED"
        error_msg = ""
        rouge1_f1: Optional[float] = None
        rouge2_f1: Optional[float] = None
        rougel_f1: Optional[float] = None
        formatted_notes: Optional[str] = None

        try:
            # 1. Validate URL & 2. Extract video ID
            t0 = time.perf_counter()
            extracted_id = extract_video_id(youtube_url)
            metrics.url_parsing_latency_seconds = round(time.perf_counter() - t0, 4)

            # 3. Retrieve transcript
            t0 = time.perf_counter()
            raw_segments = self.transcript_provider.get_transcript(extracted_id)
            metrics.transcript_retrieval_latency_seconds = round(time.perf_counter() - t0, 4)

            # 4. Preprocess transcript
            t0 = time.perf_counter()
            preprocessed = clean_transcript(raw_segments)
            clean_text = preprocessed["clean_text"]
            metrics.preprocessing_latency_seconds = round(time.perf_counter() - t0, 4)
            metrics.input_word_count = preprocessed["word_count"]

            # 5. Chunk transcript
            t0 = time.perf_counter()
            chunks = chunk_transcript(
                clean_text,
                max_words=max_words_per_chunk,
                overlap_words=overlap_words,
            )
            metrics.chunking_latency_seconds = round(time.perf_counter() - t0, 4)
            metrics.chunk_count = len(chunks)

            # 6. Generate summary
            t0 = time.perf_counter()
            notes_result = generate_notes(
                transcript=clean_text,
                client=self.llm_client,
                max_words_per_chunk=max_words_per_chunk,
                overlap_words=overlap_words,
            )
            metrics.llm_inference_latency_seconds = round(time.perf_counter() - t0, 4)

            formatted_notes = format_as_markdown(notes_result)
            validate_markdown_notes(formatted_notes)

            metrics.llm_call_count = notes_result["metadata"]["llm_call_count"]
            metrics.output_word_count = len(formatted_notes.split())

            # 10. Calculate compression ratio
            ratio, reduction = calculate_compression_ratio(
                metrics.input_word_count,
                metrics.output_word_count,
            )
            metrics.compression_ratio = ratio
            metrics.reduction_percentage = reduction

            # 12. Calculate ROUGE only when reference summaries are available
            if reference_summary:
                rouge_scores = calculate_rouge(formatted_notes, reference_summary)
                rouge1_f1 = rouge_scores["rouge-1"]["f1"]
                rouge2_f1 = rouge_scores["rouge-2"]["f1"]
                rougel_f1 = rouge_scores["rouge-l"]["f1"]

            # 11. Record success
            status = "SUCCESS"

        except Exception as e:
            error_msg = sanitize_secrets(str(e))
            status = "FAILED"
            logger.warning("Pipeline step failed for video %s: %s", extracted_id, error_msg)

        finally:
            # 7. Record total latency
            total_elapsed = round(time.perf_counter() - e2e_start, 2)
            metrics.total_processing_time_seconds = total_elapsed

        row_result = {
            "video_id": extracted_id,
            "youtube_url": youtube_url,
            "domain": domain,
            "approx_duration": approx_duration,
            "status": status,
            "transcript_words": metrics.input_word_count if status == "SUCCESS" else None,
            "summary_words": metrics.output_word_count if status == "SUCCESS" else None,
            "compression_ratio": metrics.compression_ratio if status == "SUCCESS" else None,
            "chunk_count": metrics.chunk_count if status == "SUCCESS" else None,
            "llm_calls": metrics.llm_call_count if status == "SUCCESS" else None,
            "total_latency_seconds": total_elapsed,
            "retrieval_latency": metrics.transcript_retrieval_latency_seconds,
            "preprocessing_latency": metrics.preprocessing_latency_seconds,
            "chunking_latency": metrics.chunking_latency_seconds,
            "llm_latency": metrics.llm_inference_latency_seconds,
            "rouge1_f1": rouge1_f1,
            "rouge2_f1": rouge2_f1,
            "rougel_f1": rougel_f1,
            "error": error_msg,
            "notes": notes_info,
            "metrics": metrics.to_dict(),
        }

        # Append to results.csv
        self.append_results_row(row_result)
        return row_result

    def run_dataset(
        self,
        dataset_path: Optional[Path] = None,
        results_path: Optional[Path] = None,
        limit: Optional[int] = None,
        delay_seconds: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Read test_dataset.csv and execute the benchmark on each video.

        Args:
            dataset_path: Path to dataset CSV. Defaults to evaluation/test_dataset.csv.
            results_path: Path to write results. Defaults to evaluation/results.csv.
            limit: Maximum number of videos to process to keep API usage controlled.
            delay_seconds: Cooldown between videos to respect rate limits.

        Returns:
            Tuple of (list_of_row_results, evaluation_report_dict).
        """
        ds_path = dataset_path or DEFAULT_DATASET_CSV
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found at {ds_path}")

        if results_path:
            self.results_csv = results_path
            self._ensure_csv_header(self.results_csv, RESULTS_CSV_HEADERS)

        records: List[Dict[str, str]] = []
        with open(ds_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

        if limit is not None and limit > 0:
            logger.info("Limiting benchmark run to %d videos (dataset has %d)", limit, len(records))
            records = records[:limit]

        logger.info("Executing benchmark run across %d videos", len(records))
        results: List[Dict[str, Any]] = []

        for idx, record in enumerate(records, start=1):
            vid = record.get("video_id", "unknown")
            logger.info("[%d/%d] Processing video %s...", idx, len(records), vid)
            res = self.process_dataset_record(record)
            results.append(res)

            if idx < len(records) and delay_seconds > 0:
                time.sleep(delay_seconds)

        report = generate_evaluation_report(results)
        return results, report


def generate_evaluation_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate telemetry metrics from benchmark results.

    Metrics calculated:
    - Total videos
    - Successful videos
    - Failed videos
    - Success rate (%)
    - Average latency (s)
    - Median latency (s)
    - Average compression ratio
    - Average summary length (words)
    - Average chunk count

    Args:
        results: List of processed video result dictionaries.

    Returns:
        Structured evaluation report dictionary.
    """
    total_videos = len(results)
    if total_videos == 0:
        return {
            "total_videos": 0,
            "successful_videos": 0,
            "failed_videos": 0,
            "success_rate": 0.0,
            "average_latency": 0.0,
            "median_latency": 0.0,
            "average_compression_ratio": 0.0,
            "average_summary_length": 0.0,
            "average_chunk_count": 0.0,
            # Legacy compatibility keys
            "total_runs": 0,
            "successful_runs": 0,
            "avg_latency_seconds": 0.0,
            "avg_compression_ratio": 0.0,
        }

    successful = [r for r in results if r.get("status") == "SUCCESS"]
    failed = [r for r in results if r.get("status") != "SUCCESS"]
    success_rate = round(len(successful) / total_videos * 100.0, 1)

    if not successful:
        return {
            "total_videos": total_videos,
            "successful_videos": 0,
            "failed_videos": len(failed),
            "success_rate": success_rate,
            "average_latency": 0.0,
            "median_latency": 0.0,
            "average_compression_ratio": 0.0,
            "average_summary_length": 0.0,
            "average_chunk_count": 0.0,
            # Legacy keys
            "total_runs": total_videos,
            "successful_runs": 0,
            "avg_latency_seconds": 0.0,
            "avg_compression_ratio": 0.0,
        }

    latencies = [
        r["total_latency_seconds"]
        for r in successful
        if r.get("total_latency_seconds") is not None
    ]
    ratios = [
        r["compression_ratio"]
        for r in successful
        if r.get("compression_ratio") is not None
    ]
    summary_lengths = [
        r["summary_words"]
        for r in successful
        if r.get("summary_words") is not None
    ]
    chunk_counts = [
        r["chunk_count"]
        for r in successful
        if r.get("chunk_count") is not None
    ]

    avg_latency = round(statistics.mean(latencies), 2) if latencies else 0.0
    median_latency = round(statistics.median(latencies), 2) if latencies else 0.0
    avg_compression = round(statistics.mean(ratios), 4) if ratios else 0.0
    avg_summary_len = round(statistics.mean(summary_lengths), 1) if summary_lengths else 0.0
    avg_chunks = round(statistics.mean(chunk_counts), 1) if chunk_counts else 0.0

    return {
        "total_videos": total_videos,
        "successful_videos": len(successful),
        "failed_videos": len(failed),
        "success_rate": success_rate,
        "average_latency": avg_latency,
        "median_latency": median_latency,
        "average_compression_ratio": avg_compression,
        "average_summary_length": avg_summary_len,
        "average_chunk_count": avg_chunks,
        # Legacy compatibility keys
        "total_runs": total_videos,
        "successful_runs": len(successful),
        "avg_latency_seconds": avg_latency,
        "avg_compression_ratio": avg_compression,
    }


def aggregate_benchmark_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Legacy compatibility wrapper for aggregate benchmark evaluation."""
    # Convert legacy format metrics if present
    normalized: List[Dict[str, Any]] = []
    for r in results:
        status = r.get("status", "SUCCESS")
        m = r.get("metrics", {})
        total_lat = m.get("total_processing_time_seconds", r.get("total_latency_seconds", 0.0))
        comp_ratio = m.get("compression_ratio", r.get("compression_ratio", 0.0))
        out_words = m.get("output_word_count", r.get("summary_words", 0))
        chunks = m.get("chunk_count", r.get("chunk_count", 0))

        normalized.append({
            "status": "SUCCESS" if status == "SUCCESS" else "FAILED",
            "total_latency_seconds": total_lat,
            "compression_ratio": comp_ratio,
            "summary_words": out_words,
            "chunk_count": chunks,
        })
    return generate_evaluation_report(normalized)


def format_evaluation_report(report: Dict[str, Any]) -> str:
    """Format an evaluation report dictionary into a clean Markdown table."""
    return f"""# 📊 Benchmark Evaluation Report

| Metric | Value |
| :--- | :--- |
| **Total Videos** | {report['total_videos']} |
| **Successful Videos** | {report['successful_videos']} |
| **Failed Videos** | {report['failed_videos']} |
| **Success Rate** | {report['success_rate']:.1f}% |
| **Average Latency** | {report['average_latency']:.2f}s |
| **Median Latency** | {report['median_latency']:.2f}s |
| **Average Compression Ratio** | {report['average_compression_ratio']:.4f} |
| **Average Summary Length** | {report['average_summary_length']:.1f} words |
| **Average Chunk Count** | {report['average_chunk_count']:.1f} |
"""


def main():
    """CLI Entrypoint for running benchmarks with controlled API usage."""
    parser = argparse.ArgumentParser(
        description="Run YouTube Summarizer 25-Video Benchmark Framework."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_CSV),
        help="Path to test dataset CSV (defaults to evaluation/test_dataset.csv).",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=str(DEFAULT_RESULTS_CSV),
        help="Path to output results CSV (defaults to evaluation/results.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of videos to process to keep API usage controlled.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between videos to manage rate limits.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Run benchmark against a single YouTube video ID.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Optional path to write markdown evaluation report.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    runner = BenchmarkRunner(results_csv_path=Path(args.results))

    if args.video_id:
        url = f"https://www.youtube.com/watch?v={args.video_id}"
        print(f"Benchmarking single video: {url}")
        res = runner.run_benchmark(youtube_url=url)
        print(f"Status: {res['status']}")
        print(f"Latency: {res['metrics']['total_processing_time_seconds']}s")
        print(f"Compression: {res['metrics']['compression_ratio']}")
        return

    print(f"Running benchmark on dataset: {args.dataset}")
    if args.limit:
        print(f"API Usage Protection: Limited to first {args.limit} video(s)")

    results, report = runner.run_dataset(
        dataset_path=Path(args.dataset),
        results_path=Path(args.results),
        limit=args.limit,
        delay_seconds=args.delay,
    )

    report_md = format_evaluation_report(report)
    print("\n" + report_md)

    if args.report_out:
        Path(args.report_out).write_text(report_md, encoding="utf-8")
        print(f"Report saved to {args.report_out}")


if __name__ == "__main__":
    main()
