import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from openai import APIConnectionError, APIStatusError

from potato_not_potato.benchmark import BenchmarkResult, run_benchmark
from potato_not_potato.evaluate import ITEM_FIELDS, SUMMARY_FIELDS
from potato_not_potato.models import Receipt

DEFAULT_RECEIPT_PATH = Path("data/img20260903_15395748.txt")
DEFAULT_GROUND_TRUTH_PATH = Path("data/ground-truth/img20260903_15395748.json")
DEFAULT_MODEL = "inkling-2-mini"
DEFAULT_BASE_URL = "http://localhost:8080/v1"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H:%M.%S"


def timestamped_output_path(path: Path, now: datetime | None = None) -> Path:
    """Append a YYYY-MM-DD_HH:MM.SS timestamp to an output file path stem."""
    timestamp = (now or datetime.now()).strftime(TIMESTAMP_FORMAT)
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="potato",
        description="Run a structured receipt extraction benchmark on a local LLM.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "no-key"),
        help="API key for the server (default: OPENAI_API_KEY or 'no-key')",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model identifier to request (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=DEFAULT_RECEIPT_PATH,
        help=f"Path to input receipt text file (default: {DEFAULT_RECEIPT_PATH})",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
        help=f"Path to ground-truth JSON file (default: {DEFAULT_GROUND_TRUTH_PATH})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--cache-prompt",
        action="store_true",
        help="Allow llama.cpp KV prompt caching (disabled by default to measure full prefill)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path to write full benchmark results as JSON "
            "(appends _YYYY-MM-DD_HH:MM.SS to filename)"
        ),
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=15,
        help="Maximum number of field mismatches to print (-1 for all, 0 to hide)",
    )
    return parser


def _fmt_rate(val: float | None, suffix: str = "tok/s") -> str:
    if val is None:
        return "N/A"
    return f"{val:,.2f} {suffix}"


def print_report(result: BenchmarkResult, max_mismatches: int = 15) -> None:
    perf = result.performance
    print("=" * 64)
    print("Receipt Extraction Benchmark")
    print("=" * 64)
    print(f"Model:        {result.model}")
    print(f"Endpoint:     {result.base_url}")
    print(f"Wall Time:    {perf.wall_time_seconds:.2f} s")
    print()

    print("Performance")
    print("-" * 64)
    prefill_detail = ""
    if perf.prompt_tokens is not None:
        prefill_detail = f" ({perf.prompt_tokens} tokens"
        if perf.prompt_ms is not None:
            prefill_detail += f" in {perf.prompt_ms:.1f} ms"
        if perf.cached_prompt_tokens:
            prefill_detail += f", {perf.cached_prompt_tokens} cached"
        prefill_detail += ")"

    gen_detail = ""
    if perf.completion_tokens is not None:
        gen_detail = f" ({perf.completion_tokens} tokens"
        if perf.generation_ms is not None:
            gen_detail += f" in {perf.generation_ms:.1f} ms"
        gen_detail += ")"

    print(f"Prefill:      {_fmt_rate(perf.prefill_tokens_per_sec)}{prefill_detail}")
    print(f"Generation:   {_fmt_rate(perf.generation_tokens_per_sec)}{gen_detail}")
    if perf.timings_source != "llama.cpp":
        print(
            "Note: Server did not return llama.cpp `timings`; "
            "prefill tok/s unavailable and generation tok/s uses total wall time."
        )
    print()

    if result.error or result.accuracy is None:
        print("Accuracy")
        print("-" * 64)
        print(f"Error: {result.error}")
        return

    acc = result.accuracy
    print("Accuracy (Exact Match)")
    print("-" * 64)
    print(
        f"Overall Field Accuracy: {acc.overall_accuracy * 100:6.2f}% "
        f"({acc.matched_fields}/{acc.total_fields} fields)"
    )
    print(
        f"Exact Item Match:       {acc.item_exact_rate * 100:6.2f}% "
        f"({acc.exact_items}/{max(acc.expected_items, acc.predicted_items)} items; "
        f"predicted {acc.predicted_items}, expected {acc.expected_items})"
    )
    print()
    print("Per-Field Accuracy:")
    for field in ITEM_FIELDS:
        pct = acc.field_accuracies[field] * 100
        print(f"  item.{field:<12} {pct:6.2f}%")
    for field in SUMMARY_FIELDS:
        status = "MATCH" if acc.field_accuracies[field] == 1.0 else "MISMATCH"
        print(f"  receipt.{field:<9} {status}")

    if acc.mismatches and max_mismatches != 0:
        print()
        shown = (
            acc.mismatches if max_mismatches < 0 else acc.mismatches[:max_mismatches]
        )
        print(f"Mismatches ({len(shown)} of {len(acc.mismatches)} shown):")
        for m in shown:
            print(
                f"  - {m.location}.{m.field}: "
                f"expected={m.expected!r}, actual={m.actual!r}"
            )
        remaining = len(acc.mismatches) - len(shown)
        if remaining > 0:
            print(f"  ... and {remaining} more (pass --max-mismatches -1 to show all)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.receipt.is_file():
        print(f"Error: Receipt file not found: {args.receipt}", file=sys.stderr)
        return 1
    if not args.ground_truth.is_file():
        print(
            f"Error: Ground-truth file not found: {args.ground_truth}",
            file=sys.stderr,
        )
        return 1

    receipt_text = args.receipt.read_text(encoding="utf-8")
    ground_truth = Receipt.model_validate_json(
        args.ground_truth.read_text(encoding="utf-8")
    )

    try:
        result = run_benchmark(
            receipt_text=receipt_text,
            ground_truth=ground_truth,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            cache_prompt=args.cache_prompt,
        )
    except APIConnectionError as exc:
        print(
            f"Error: Could not connect to LLM server at {args.base_url} ({exc}).\n"
            "Make sure llama-server is running and accessible.",
            file=sys.stderr,
        )
        return 1
    except APIStatusError as exc:
        print(
            f"Error: Server returned HTTP {exc.status_code}: {exc.message}",
            file=sys.stderr,
        )
        return 1

    print_report(result, max_mismatches=args.max_mismatches)

    if args.output is not None:
        output_path = timestamped_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved full benchmark results to {output_path}")

    return 0 if result.error is None else 1
