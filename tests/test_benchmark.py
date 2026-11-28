import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from openai.types.chat import ParsedChatCompletion

from potato_not_potato.benchmark import BenchmarkResult, PerformanceMetrics
from potato_not_potato.cli import format_markdown_report, timestamped_output_paths
from potato_not_potato.cli import main as cli_main
from potato_not_potato.evaluate import evaluate_receipt
from potato_not_potato.models import Receipt, ReceiptItem

GROUND_TRUTH_PATH = Path("data/ground-truth/img20260903_15395748.json")


class ReceiptBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ground_truth = Receipt.model_validate_json(
            GROUND_TRUTH_PATH.read_text(encoding="utf-8")
        )

    def test_exact_match_ground_truth_scores_100_percent(self) -> None:
        metrics = evaluate_receipt(self.ground_truth, self.ground_truth)
        self.assertEqual(metrics.overall_accuracy, 1.0)
        self.assertEqual(metrics.item_exact_rate, 1.0)
        self.assertEqual(metrics.exact_items, 43)
        self.assertEqual(metrics.mismatches, [])

    def test_missing_middle_item_aligns_remaining_items(self) -> None:
        items_missing_one = [
            item.model_copy()
            for idx, item in enumerate(self.ground_truth.items)
            if idx != 10
        ]
        predicted = Receipt(
            items=items_missing_one,
            subtotal=self.ground_truth.subtotal,
            tax=self.ground_truth.tax,
            total=self.ground_truth.total,
        )
        metrics = evaluate_receipt(predicted, self.ground_truth)
        self.assertEqual(metrics.exact_items, 42)
        self.assertEqual(len(metrics.mismatches), 1)
        self.assertEqual(metrics.mismatches[0].location, "items[10]")
        self.assertEqual(metrics.mismatches[0].field, "<missing_item>")

    def test_field_mismatch_and_extra_item_penalized(self) -> None:
        modified_items = [item.model_copy() for item in self.ground_truth.items]
        modified_items[3] = ReceiptItem(
            name="NY MUSTARD DIJON ORG 8 OZ",  # 'NY' instead of 'NV'
            quantity=modified_items[3].quantity,
            unit_price=modified_items[3].unit_price,
            price=modified_items[3].price,
        )
        modified_items.append(
            ReceiptItem(name="SPURIOUS ITEM", quantity=None, unit_price=None, price=1.0)
        )
        predicted = Receipt(
            items=modified_items,
            subtotal=self.ground_truth.subtotal,
            tax=self.ground_truth.tax,
            total=999.99,
        )
        metrics = evaluate_receipt(predicted, self.ground_truth)
        self.assertEqual(metrics.exact_items, 42)
        self.assertEqual(metrics.predicted_items, 44)
        # Denominator uses max(43, 44) * 4 + 3 = 179
        self.assertEqual(metrics.total_fields, 44 * 4 + 3)
        # Matched: 42*4 + 3 (from item[3]) + 2 (subtotal, tax) = 173
        self.assertEqual(metrics.matched_fields, 173)

    def test_timestamped_output_paths_ignores_extension_and_uses_stem(self) -> None:
        fixed_now = datetime(2026, 11, 27, 13, 5, 2)
        self.assertEqual(
            timestamped_output_paths(Path("/tmp/dir/result.json"), now=fixed_now),
            (
                Path("/tmp/dir/result_2026-11-27_13:05.02.json"),
                Path("/tmp/dir/result_2026-11-27_13:05.02.md"),
            ),
        )
        self.assertEqual(
            timestamped_output_paths(Path("/tmp/dir/result.md"), now=fixed_now),
            (
                Path("/tmp/dir/result_2026-11-27_13:05.02.json"),
                Path("/tmp/dir/result_2026-11-27_13:05.02.md"),
            ),
        )
        self.assertEqual(
            timestamped_output_paths(Path("my.results.json"), now=fixed_now),
            (
                Path("my.results_2026-11-27_13:05.02.json"),
                Path("my.results_2026-11-27_13:05.02.md"),
            ),
        )
        self.assertEqual(
            timestamped_output_paths(Path("results"), now=fixed_now),
            (
                Path("results_2026-11-27_13:05.02.json"),
                Path("results_2026-11-27_13:05.02.md"),
            ),
        )

    def _make_fake_response(self, receipt: Receipt) -> ParsedChatCompletion[Receipt]:
        return ParsedChatCompletion[Receipt].model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "inkling-2-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": receipt.model_dump_json(),
                            "parsed": receipt,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1120,
                    "completion_tokens": 980,
                    "total_tokens": 2100,
                },
                "timings": {
                    "cache_n": 0,
                    "prompt_n": 1120,
                    "prompt_ms": 224.0,
                    "prompt_per_second": 5000.0,
                    "predicted_n": 980,
                    "predicted_ms": 8000.0,
                    "predicted_per_second": 122.5,
                },
            }
        )

    def test_cli_end_to_end_with_llamacpp_timings(self) -> None:
        fake_response = self._make_fake_response(self.ground_truth)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("potato_not_potato.benchmark.OpenAI") as mock_openai_cls,
        ):
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.parse.return_value = fake_response
            out_path = Path(tmpdir) / "result.json"

            exit_code = cli_main(["--output", str(out_path)])
            self.assertEqual(exit_code, 0)
            self.assertFalse(out_path.exists())

            written_files = sorted(Path(tmpdir).iterdir(), key=lambda p: p.suffix)
            self.assertEqual(len(written_files), 2)
            json_file, md_file = written_files
            self.assertEqual(json_file.stem, md_file.stem)
            self.assertRegex(
                json_file.stem,
                r"^result_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}\.\d{2}$",
            )
            self.assertEqual(json_file.suffix, ".json")
            self.assertEqual(md_file.suffix, ".md")

            saved = json.loads(json_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["model"], "inkling-2-mini")
            self.assertEqual(saved["performance"]["prefill_tokens_per_sec"], 5000.0)
            self.assertEqual(saved["performance"]["generation_tokens_per_sec"], 122.5)
            self.assertEqual(saved["performance"]["timings_source"], "llama.cpp")
            self.assertEqual(saved["accuracy"]["overall_accuracy"], 1.0)
            self.assertEqual(saved["accuracy"]["exact_items"], 43)

            md_text = md_file.read_text(encoding="utf-8")
            self.assertIn("`inkling-2-mini`", md_text)
            self.assertIn("`http://localhost:8080/v1`", md_text)
            self.assertIn("5,000.00 tok/s", md_text)
            self.assertIn("122.50 tok/s", md_text)
            self.assertIn("100.00% (175/175)", md_text)
            self.assertIn("100.00% (43/43)", md_text)
            self.assertIn("FD BROTH CHCKN LOW SDM 32 OZ", md_text)
            self.assertIn("229.23", md_text)
            self.assertIn("## Raw Content", md_text)

    def test_cli_ignores_extension_and_writes_all_mismatches_to_markdown(self) -> None:
        modified_items = [item.model_copy() for item in self.ground_truth.items]
        modified_items[0] = ReceiptItem(
            name="WRONG ITEM 0",
            quantity=modified_items[0].quantity,
            unit_price=modified_items[0].unit_price,
            price=modified_items[0].price,
        )
        modified_items[1] = ReceiptItem(
            name="WRONG ITEM 1",
            quantity=modified_items[1].quantity,
            unit_price=modified_items[1].unit_price,
            price=modified_items[1].price,
        )
        imperfect = Receipt(
            items=modified_items,
            subtotal=self.ground_truth.subtotal,
            tax=self.ground_truth.tax,
            total=self.ground_truth.total,
        )
        fake_response = self._make_fake_response(imperfect)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("potato_not_potato.benchmark.OpenAI") as mock_openai_cls,
        ):
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.parse.return_value = fake_response
            out_path = Path(tmpdir) / "custom_stem.txt"

            exit_code = cli_main(["--output", str(out_path), "--max-mismatches", "1"])
            self.assertEqual(exit_code, 0)
            self.assertFalse(out_path.exists())

            written_files = sorted(Path(tmpdir).iterdir(), key=lambda p: p.suffix)
            self.assertEqual(len(written_files), 2)
            json_file, md_file = written_files
            self.assertEqual(json_file.stem, md_file.stem)
            self.assertRegex(
                json_file.stem,
                r"^custom_stem_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}\.\d{2}$",
            )

            md_text = md_file.read_text(encoding="utf-8")
            self.assertIn("### Mismatches (2)", md_text)
            self.assertIn("WRONG ITEM 0", md_text)
            self.assertIn("WRONG ITEM 1", md_text)

    def test_format_markdown_report_handles_mismatches_and_pipes(self) -> None:
        modified_items = [
            item.model_copy()
            for idx, item in enumerate(self.ground_truth.items)
            if idx != 10
        ]
        modified_items[3] = ReceiptItem(
            name="NY | MUSTARD `DIJON` ORG",
            quantity=modified_items[3].quantity,
            unit_price=modified_items[3].unit_price,
            price=modified_items[3].price,
        )
        modified_items.append(
            ReceiptItem(name="EXTRA | ITEM", quantity=None, unit_price=None, price=1.0)
        )
        predicted = Receipt(
            items=modified_items,
            subtotal=self.ground_truth.subtotal,
            tax=self.ground_truth.tax,
            total=999.99,
        )
        metrics = evaluate_receipt(predicted, self.ground_truth)
        result = BenchmarkResult(
            model="test-model",
            base_url="http://localhost:8080/v1",
            performance=PerformanceMetrics(
                wall_time_seconds=1.25,
                prompt_tokens=100,
                cached_prompt_tokens=0,
                completion_tokens=200,
                prompt_ms=50.0,
                generation_ms=1200.0,
                prefill_tokens_per_sec=2000.0,
                generation_tokens_per_sec=166.67,
                timings_source="llama.cpp",
            ),
            accuracy=metrics,
            predicted=predicted,
            raw_content=predicted.model_dump_json(),
            error=None,
        )

        md = format_markdown_report(result)
        self.assertIn("- **Error**: None", md)
        self.assertIn("NY \\| MUSTARD `DIJON` ORG", md)
        self.assertIn("`<missing_item>`", md)
        self.assertIn("`<extra_item>`", md)
        self.assertIn("EXTRA \\| ITEM", md)
        self.assertIn("`receipt.total` | 0.00% (MISMATCH)", md)

    def test_format_markdown_report_handles_error_result(self) -> None:
        result = BenchmarkResult(
            model="test-model",
            base_url="http://localhost:8080/v1",
            performance=PerformanceMetrics(
                wall_time_seconds=0.5,
                prompt_tokens=None,
                cached_prompt_tokens=None,
                completion_tokens=None,
                prompt_ms=None,
                generation_ms=None,
                prefill_tokens_per_sec=None,
                generation_tokens_per_sec=None,
                timings_source="wall_clock_fallback",
            ),
            accuracy=None,
            predicted=None,
            raw_content=None,
            error="Validation failed\non line 2",
        )

        md = format_markdown_report(result)
        self.assertIn("- **Error**: Yes (see error details below)", md)
        self.assertIn("## Error", md)
        self.assertIn("Validation failed\non line 2", md)
        self.assertIn("_Accuracy metrics unavailable._", md)
        self.assertIn("_No predicted receipt._", md)
        self.assertIn("_None_", md)


if __name__ == "__main__":
    unittest.main()
