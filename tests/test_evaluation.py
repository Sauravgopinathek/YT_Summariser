"""Unit tests for evaluation.metrics and evaluation.benchmark."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from evaluation.benchmark import BenchmarkRunner, aggregate_benchmark_report
from evaluation.metrics import (
    PerformanceMetrics,
    calculate_compression_ratio,
    calculate_rouge,
)
from modules.llm import NVIDIAClient


class TestEvaluationAndMetrics(unittest.TestCase):
    """Test suite for metrics calculation, ROUGE scoring, and benchmark logging."""

    def test_calculate_compression_ratio(self):
        """Test calculation of compression ratio and percentage reduction."""
        ratio, reduction = calculate_compression_ratio(transcript_words=1000, summary_words=200)
        self.assertEqual(ratio, 0.2)
        self.assertEqual(reduction, 80.0)

        # Zero input words
        r0, red0 = calculate_compression_ratio(transcript_words=0, summary_words=50)
        self.assertEqual(r0, 0.0)
        self.assertEqual(red0, 0.0)

    def test_calculate_rouge_identical_text(self):
        """Test ROUGE scoring on identical strings yields 1.0."""
        text = "Deep learning models require gradient descent optimization on GPU hardware."
        scores = calculate_rouge(candidate_summary=text, reference_summary=text)

        self.assertEqual(scores["rouge-1"]["f1"], 1.0)
        self.assertEqual(scores["rouge-2"]["f1"], 1.0)
        self.assertEqual(scores["rouge-l"]["f1"], 1.0)

    def test_calculate_rouge_partial_overlap(self):
        """Test ROUGE scoring on partially overlapping texts."""
        candidate = "Neural networks optimize loss using gradient descent."
        reference = "Neural networks minimize objective functions using gradient descent."
        scores = calculate_rouge(candidate, reference)

        self.assertGreater(scores["rouge-1"]["f1"], 0.5)
        self.assertLess(scores["rouge-1"]["f1"], 1.0)
        self.assertIn("rouge-l", scores)

    def test_calculate_rouge_empty_input_raises_error(self):
        """Test that calculating ROUGE with empty input raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_rouge("", "Valid reference summary.")
        with self.assertRaises(ValueError):
            calculate_rouge("Valid candidate summary.", "   ")

    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics initialization and dictionary serialization."""
        metrics = PerformanceMetrics(
            url_parsing_latency_seconds=0.001,
            input_word_count=500,
            output_word_count=100,
            compression_ratio=0.2,
        )
        d = metrics.to_dict()
        self.assertEqual(d["url_parsing_latency_seconds"], 0.001)
        self.assertEqual(d["input_word_count"], 500)
        self.assertEqual(d["output_word_count"], 100)

    @patch("evaluation.benchmark.YouTubeTranscriptProvider")
    def test_benchmark_runner_execution_and_csv_generation(self, mock_provider_cls):
        """Test BenchmarkRunner measures all pipeline stages and writes CSV row."""
        # Setup mock transcript segments
        mock_provider = MagicMock()
        mock_provider.get_transcript.return_value = [
            {"text": "Sentence one explains backpropagation.", "start": 0.0, "duration": 2.0},
            {"text": "Sentence two explains gradient descent.", "start": 2.0, "duration": 2.0},
        ]
        mock_provider_cls.return_value = mock_provider

        # Setup mock LLM client
        mock_client = MagicMock(spec=NVIDIAClient)
        mock_client.generate.return_value = (
            "# Executive Summary\n"
            "This video explains neural backpropagation and gradient descent.\n\n"
            "# Key Concepts\n"
            "- Backpropagation mechanism.\n\n"
            "# Action Items\n"
            "- Implement gradient descent update rule."
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_csv = Path(tmp_dir) / "test_benchmark.csv"
            runner = BenchmarkRunner(output_csv_path=test_csv, llm_client=mock_client)

            result = runner.run_benchmark(
                youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                category="machine-learning",
                reference_summary="This lecture covers neural backpropagation and gradient optimization.",
            )

            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["video_id"], "dQw4w9WgXcQ")
            self.assertIn("rouge-1", result["rouge"])
            self.assertGreater(result["metrics"]["input_word_count"], 0)
            self.assertGreater(result["metrics"]["output_word_count"], 0)

            # Verify CSV file existence and headers
            self.assertTrue(test_csv.exists())
            lines = test_csv.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 2)  # header + row
            self.assertIn("dQw4w9WgXcQ", lines[1])
            self.assertIn("machine-learning", lines[1])

    def test_aggregate_benchmark_report(self):
        """Test aggregation calculation across multiple benchmark runs."""
        sample_results = [
            {
                "status": "SUCCESS",
                "metrics": {"total_processing_time_seconds": 2.5, "compression_ratio": 0.20},
            },
            {
                "status": "SUCCESS",
                "metrics": {"total_processing_time_seconds": 3.5, "compression_ratio": 0.30},
            },
            {
                "status": "FAILED: TranscriptError",
                "metrics": {"total_processing_time_seconds": 0.5, "compression_ratio": 0.0},
            },
        ]
        report = aggregate_benchmark_report(sample_results)
        self.assertEqual(report["total_runs"], 3)
        self.assertEqual(report["successful_runs"], 2)
        self.assertEqual(report["success_rate"], 66.7)
        self.assertEqual(report["avg_latency_seconds"], 3.0)
        self.assertEqual(report["avg_compression_ratio"], 0.25)

    def test_test_dataset_csv_structure_and_schema(self):
        """Test evaluation/test_dataset.csv exists, has 25 videos, and valid schema."""
        import csv
        from evaluation.benchmark import DEFAULT_DATASET_CSV
        from modules.url_parser import validate_video_id

        self.assertTrue(DEFAULT_DATASET_CSV.exists())
        with open(DEFAULT_DATASET_CSV, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected_headers = {"video_id", "youtube_url", "domain", "approx_duration", "reference_summary", "notes"}
            self.assertEqual(set(reader.fieldnames or []), expected_headers)
            rows = list(reader)

        self.assertEqual(len(rows), 25, "Dataset must contain exactly 25 video records")
        for row in rows:
            self.assertTrue(validate_video_id(row["video_id"]), f"Invalid video_id: {row['video_id']}")
            self.assertIn(row["video_id"], row["youtube_url"])
            self.assertTrue(row["domain"].strip())

    def test_process_dataset_record_and_results_csv(self):
        """Test processing dataset record produces results.csv row with all metrics."""
        import csv
        from evaluation.benchmark import BenchmarkRunner, RESULTS_CSV_HEADERS

        mock_provider = MagicMock()
        mock_provider.get_transcript.return_value = [
            {"text": "Understanding calculus and linear algebra fundamentals.", "start": 0.0, "duration": 3.0},
            {"text": "Matrices map vectors to different coordinate spaces.", "start": 3.0, "duration": 3.0},
        ]

        mock_client = MagicMock(spec=NVIDIAClient)
        mock_client.generate.return_value = (
            "# Executive Summary\n"
            "This video covers linear algebra and matrix transformations.\n\n"
            "# Key Concepts\n"
            "- Matrices transform vectors into new spaces.\n\n"
            "# Action Items\n"
            "- Practice matrix multiplication."
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            res_csv = Path(tmp_dir) / "results.csv"
            runner = BenchmarkRunner(
                llm_client=mock_client,
                transcript_provider=mock_provider,
                results_csv_path=res_csv,
            )

            record = {
                "video_id": "fNk_zzaMoSs",
                "youtube_url": "https://www.youtube.com/watch?v=fNk_zzaMoSs",
                "domain": "Mathematics",
                "approx_duration": "10m",
                "reference_summary": "This video covers linear algebra and transformations.",
                "notes": "Essence of linear algebra",
            }

            res = runner.process_dataset_record(record)

            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["video_id"], "fNk_zzaMoSs")
            self.assertIsNotNone(res["rouge1_f1"])
            self.assertGreater(res["transcript_words"], 0)
            self.assertGreater(res["summary_words"], 0)

            # Check CSV written
            self.assertTrue(res_csv.exists())
            with open(res_csv, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, RESULTS_CSV_HEADERS)
                rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["video_id"], "fNk_zzaMoSs")
                self.assertEqual(rows[0]["status"], "SUCCESS")
                self.assertNotEqual(rows[0]["rouge1_f1"], "")

    def test_process_dataset_record_without_reference_summary_omits_rouge(self):
        """Test that omitting reference summary does not fabricate ROUGE scores."""
        import csv
        from evaluation.benchmark import BenchmarkRunner

        mock_provider = MagicMock()
        mock_provider.get_transcript.return_value = [
            {"text": "Content without reference summary.", "start": 0.0, "duration": 2.0}
        ]

        mock_client = MagicMock(spec=NVIDIAClient)
        mock_client.generate.return_value = (
            "# Executive Summary\nSummary text.\n\n# Key Concepts\n- Concept.\n\n# Action Items\n- Action."
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            res_csv = Path(tmp_dir) / "results.csv"
            runner = BenchmarkRunner(
                llm_client=mock_client,
                transcript_provider=mock_provider,
                results_csv_path=res_csv,
            )

            record = {
                "video_id": "bBC-nXj3Ng4",
                "youtube_url": "https://www.youtube.com/watch?v=bBC-nXj3Ng4",
                "domain": "Computer Science",
                "approx_duration": "10m",
                "reference_summary": "",
                "notes": "No ref summary",
            }

            res = runner.process_dataset_record(record)
            self.assertEqual(res["status"], "SUCCESS")
            self.assertIsNone(res["rouge1_f1"])
            self.assertIsNone(res["rouge2_f1"])
            self.assertIsNone(res["rougel_f1"])

            with open(res_csv, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(rows[0]["rouge1_f1"], "")
                self.assertEqual(rows[0]["rouge2_f1"], "")
                self.assertEqual(rows[0]["rougel_f1"], "")

    def test_process_dataset_record_failure_handling(self):
        """Test that failure in a single video record is handled gracefully without crashing."""
        from evaluation.benchmark import BenchmarkRunner
        from modules.exceptions import CaptionsDisabledError

        mock_provider = MagicMock()
        mock_provider.get_transcript.side_effect = CaptionsDisabledError("Subtitles disabled")

        with tempfile.TemporaryDirectory() as tmp_dir:
            res_csv = Path(tmp_dir) / "results.csv"
            runner = BenchmarkRunner(
                transcript_provider=mock_provider,
                results_csv_path=res_csv,
            )

            record = {
                "video_id": "invalid0000",
                "youtube_url": "https://www.youtube.com/watch?v=invalid0000",
                "domain": "General",
                "approx_duration": "5m",
                "reference_summary": "",
                "notes": "Testing failure handling",
            }

            res = runner.process_dataset_record(record)
            self.assertEqual(res["status"], "FAILED")
            self.assertIn("Subtitles disabled", res["error"])
            self.assertIsNone(res["transcript_words"])
            self.assertIsNone(res["summary_words"])

    def test_generate_evaluation_report_metrics(self):
        """Test generation of all required aggregate evaluation metrics."""
        from evaluation.benchmark import generate_evaluation_report, format_evaluation_report

        mock_results = [
            {
                "status": "SUCCESS",
                "total_latency_seconds": 4.0,
                "compression_ratio": 0.20,
                "summary_words": 200,
                "chunk_count": 2,
            },
            {
                "status": "SUCCESS",
                "total_latency_seconds": 6.0,
                "compression_ratio": 0.30,
                "summary_words": 300,
                "chunk_count": 4,
            },
            {
                "status": "FAILED",
                "total_latency_seconds": 1.0,
                "compression_ratio": None,
                "summary_words": None,
                "chunk_count": None,
            },
        ]

        report = generate_evaluation_report(mock_results)

        self.assertEqual(report["total_videos"], 3)
        self.assertEqual(report["successful_videos"], 2)
        self.assertEqual(report["failed_videos"], 1)
        self.assertEqual(report["success_rate"], 66.7)
        self.assertEqual(report["average_latency"], 5.0)
        self.assertEqual(report["median_latency"], 5.0)
        self.assertEqual(report["average_compression_ratio"], 0.25)
        self.assertEqual(report["average_summary_length"], 250.0)
        self.assertEqual(report["average_chunk_count"], 3.0)

        # Verify formatted markdown
        md = format_evaluation_report(report)
        self.assertIn("66.7%", md)
        self.assertIn("Total Videos", md)
        self.assertIn("Median Latency", md)

    def test_human_evaluation_template_csv_schema(self):
        """Test human_evaluation_template.csv exists, has 25 rows, and valid columns."""
        import csv
        from evaluation.human_eval import DEFAULT_TEMPLATE_PATH, EVALUATION_COLUMNS

        self.assertTrue(DEFAULT_TEMPLATE_PATH.exists())
        with open(DEFAULT_TEMPLATE_PATH, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, EVALUATION_COLUMNS)
            rows = list(reader)

        self.assertEqual(len(rows), 25)
        for r in rows:
            self.assertTrue(r["video_id"].strip())
            # Must NOT contain fabricated scores
            self.assertEqual(r["completeness_score"], "")
            self.assertEqual(r["scannability_score"], "")
            self.assertEqual(r["fact_verification_score"], "")

    def test_load_human_evaluations_skips_empty_template(self):
        """Test loading unfilled template returns empty list without errors."""
        from evaluation.human_eval import load_human_evaluations, DEFAULT_TEMPLATE_PATH
        records = load_human_evaluations(DEFAULT_TEMPLATE_PATH)
        self.assertEqual(records, [])

    def test_load_human_evaluations_validates_scores(self):
        """Test loading valid evaluation rows parses scores and enforces [1, 5] bounds."""
        import csv
        from evaluation.human_eval import load_human_evaluations

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "eval.csv"
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["video_id", "evaluator_id", "completeness_score", "scannability_score", "fact_verification_score", "comments"])
                writer.writerow(["aircAruvnKk", "eval_1", "5", "4", "5", "Excellent clarity"])
                writer.writerow(["aircAruvnKk", "eval_2", "4", "4", "4", "Good"])
                writer.writerow(["bBC-nXj3Ng4", "eval_1", "4", "5", "5", "Accurate terms"])

            records = load_human_evaluations(csv_path)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["completeness_score"], 5.0)

            # Test invalid score out of bounds
            bad_csv = Path(tmp_dir) / "bad_eval.csv"
            with open(bad_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["video_id", "evaluator_id", "completeness_score", "scannability_score", "fact_verification_score", "comments"])
                writer.writerow(["aircAruvnKk", "eval_1", "6", "4", "5", "Too high"])

            with self.assertRaises(ValueError):
                load_human_evaluations(bad_csv)

    def test_calculate_human_eval_metrics_and_formatting(self):
        """Test overall and per-video averages calculation and report formatting."""
        from evaluation.human_eval import calculate_human_eval_metrics, format_human_eval_report

        records = [
            {
                "video_id": "vid1",
                "evaluator_id": "r1",
                "completeness_score": 4.0,
                "scannability_score": 5.0,
                "fact_verification_score": 5.0,
                "comments": "",
            },
            {
                "video_id": "vid1",
                "evaluator_id": "r2",
                "completeness_score": 5.0,
                "scannability_score": 5.0,
                "fact_verification_score": 4.0,
                "comments": "",
            },
            {
                "video_id": "vid2",
                "evaluator_id": "r1",
                "completeness_score": 3.0,
                "scannability_score": 4.0,
                "fact_verification_score": 4.0,
                "comments": "",
            },
        ]

        metrics = calculate_human_eval_metrics(records)

        self.assertEqual(metrics["total_evaluations"], 3)
        self.assertEqual(metrics["unique_videos"], 2)
        self.assertEqual(metrics["unique_evaluators"], 2)
        # Completeness: (4 + 5 + 3) / 3 = 4.0
        self.assertEqual(metrics["average_completeness"], 4.0)
        # Scannability: (5 + 5 + 4) / 3 = 4.67
        self.assertEqual(metrics["average_scannability"], 4.67)
        # Fact verification: (5 + 4 + 4) / 3 = 4.33
        self.assertEqual(metrics["average_fact_verification"], 4.33)

        # Per-video metrics
        self.assertIn("vid1", metrics["per_video"])
        self.assertEqual(metrics["per_video"]["vid1"]["evaluation_count"], 2)
        self.assertEqual(metrics["per_video"]["vid1"]["average_completeness"], 4.5)
        self.assertEqual(metrics["per_video"]["vid1"]["average_scannability"], 5.0)

        # Markdown report formatting
        md = format_human_eval_report(metrics)
        self.assertIn("vid1", md)
        self.assertIn("vid2", md)
        self.assertIn("Average Completeness", md)
        self.assertIn("Average Scannability", md)
        self.assertIn("Average Fact Verification", md)


if __name__ == "__main__":
    unittest.main()


