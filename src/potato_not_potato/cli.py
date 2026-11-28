import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError

from potato_not_potato.benchmark import BenchmarkResult, run_benchmark
from potato_not_potato.evaluate import ITEM_FIELDS, SUMMARY_FIELDS
from potato_not_potato.models import Receipt

DEFAULT_RECEIPT_PATH = Path("data/img20260903_15395748.txt")
DEFAULT_GROUND_TRUTH_PATH = Path("data/ground-truth/img20260903_15395748.json")
DEFAULT_MODEL = "inkling-2-mini"
DEFAULT_BASE_URL = "http://localhost:8080/v1"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H:%M.%S"


def timestamped_output_paths(
    path: Path, now: datetime | None = None
) -> tuple[Path, Path]:
    """Return timestamped (.json, .md) output paths using `path.stem`."""
    timestamp = (now or datetime.now()).strftime(TIMESTAMP_FORMAT)
    base_name = f"{path.stem}_{timestamp}"
    return (
        path.with_name(f"{base_name}.json"),
        path.with_name(f"{base_name}.md"),
    )


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
            "Optional output path/prefix to write full benchmark results as JSON "
            "and Markdown (ignores extension and writes "
            "<stem>_YYYY-MM-DD_HH:MM.SS.{json,md})"
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


def _fmt_int(val: int | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:,}"


def _fmt_ms(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:,.2f} ms"


def _fmt_opt_float(val: float | None) -> str:
    if val is None:
        return "null"
    return str(val)


def _md_escape_cell(text: str) -> str:
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
        .replace("|", "\\|")
    )


def _md_code_span(text: str, *, escape_pipes: bool = False) -> str:
    normalized = text.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")
    if escape_pipes:
        normalized = normalized.replace("|", "\\|")
    if not normalized:
        return '""'
    runs = re.findall(r"`+", normalized)
    fence = "`" * (max((len(r) for r in runs), default=0) + 1)
    needs_pad = (
        normalized.startswith("`")
        or normalized.endswith("`")
        or (
            len(normalized) >= 2
            and normalized.startswith(" ")
            and normalized.endswith(" ")
        )
    )
    if needs_pad:
        return f"{fence} {normalized} {fence}"
    return f"{fence}{normalized}{fence}"


def _md_json_cell(val: Any) -> str:
    return _md_code_span(json.dumps(val), escape_pipes=True)


def _md_fenced_block(content: str, info: str = "") -> str:
    runs = re.findall(r"`+", content)
    fence = "`" * max(3, max((len(r) for r in runs), default=0) + 1)
    body = content if content.endswith("\n") else f"{content}\n"
    return f"{fence}{info}\n{body}{fence}"


def format_markdown_report(result: BenchmarkResult) -> str:
    """Format a BenchmarkResult as a complete GitHub-Flavored Markdown report."""
    perf = result.performance
    lines: list[str] = [
        "# Receipt Extraction Benchmark Report",
        "",
        f"- **Model**: {_md_code_span(result.model)}",
        f"- **Base URL**: {_md_code_span(result.base_url)}",
    ]
    if result.error is None:
        lines.append("- **Error**: None")
    else:
        lines.extend(
            [
                "- **Error**: Yes (see error details below)",
                "",
                "## Error",
                "",
                _md_fenced_block(result.error, "text"),
            ]
        )

    lines.extend(
        [
            "",
            "## Performance",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Wall Time (`wall_time_seconds`) | {perf.wall_time_seconds:.2f} s |",
            f"| Prompt Tokens (`prompt_tokens`) | {_fmt_int(perf.prompt_tokens)} |",
            (
                "| Cached Prompt Tokens (`cached_prompt_tokens`) | "
                f"{_fmt_int(perf.cached_prompt_tokens)} |"
            ),
            (
                "| Completion Tokens (`completion_tokens`) | "
                f"{_fmt_int(perf.completion_tokens)} |"
            ),
            f"| Prompt Time (`prompt_ms`) | {_fmt_ms(perf.prompt_ms)} |",
            f"| Generation Time (`generation_ms`) | {_fmt_ms(perf.generation_ms)} |",
            (
                "| Prefill Throughput (`prefill_tokens_per_sec`) | "
                f"{_fmt_rate(perf.prefill_tokens_per_sec)} |"
            ),
            (
                "| Generation Throughput (`generation_tokens_per_sec`) | "
                f"{_fmt_rate(perf.generation_tokens_per_sec)} |"
            ),
            (
                "| Timings Source (`timings_source`) | "
                f"{_md_code_span(perf.timings_source, escape_pipes=True)} |"
            ),
            "",
            "## Accuracy",
            "",
        ]
    )

    if result.accuracy is None:
        lines.append("_Accuracy metrics unavailable._")
    else:
        acc = result.accuracy
        item_denom = max(acc.expected_items, acc.predicted_items)
        lines.extend(
            [
                "| Metric | Value |",
                "| --- | --- |",
                (
                    "| Overall Field Accuracy (`overall_accuracy`) | "
                    f"{acc.overall_accuracy * 100:.2f}% "
                    f"({acc.matched_fields}/{acc.total_fields}) |"
                ),
                f"| Matched Fields (`matched_fields`) | {acc.matched_fields:,} |",
                f"| Total Fields (`total_fields`) | {acc.total_fields:,} |",
                (
                    "| Exact Item Match Rate (`item_exact_rate`) | "
                    f"{acc.item_exact_rate * 100:.2f}% "
                    f"({acc.exact_items}/{item_denom}) |"
                ),
                f"| Exact Items (`exact_items`) | {acc.exact_items:,} |",
                f"| Expected Items (`expected_items`) | {acc.expected_items:,} |",
                f"| Predicted Items (`predicted_items`) | {acc.predicted_items:,} |",
                "",
                "### Per-Field Accuracy",
                "",
                "| Field | Accuracy |",
                "| --- | --- |",
            ]
        )
        for field in ITEM_FIELDS:
            pct = acc.field_accuracies[field] * 100
            lines.append(f"| `item.{field}` | {pct:.2f}% |")
        for field in SUMMARY_FIELDS:
            pct = acc.field_accuracies[field] * 100
            status = "MATCH" if acc.field_accuracies[field] == 1.0 else "MISMATCH"
            lines.append(f"| `receipt.{field}` | {pct:.2f}% ({status}) |")

        lines.extend(["", f"### Mismatches ({len(acc.mismatches)})", ""])
        if not acc.mismatches:
            lines.append("_No mismatches._")
        else:
            lines.extend(
                [
                    "| Location | Field | Expected | Actual |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for m in acc.mismatches:
                loc_cell = _md_code_span(m.location, escape_pipes=True)
                field_cell = _md_code_span(m.field, escape_pipes=True)
                exp_cell = _md_json_cell(m.expected)
                act_cell = _md_json_cell(m.actual)
                lines.append(f"| {loc_cell} | {field_cell} | {exp_cell} | {act_cell} |")

    lines.extend(["", "## Predicted Receipt", ""])
    if result.predicted is None:
        lines.append("_No predicted receipt._")
    else:
        pred = result.predicted
        lines.extend([f"### Items ({len(pred.items)})", ""])
        if not pred.items:
            lines.append("_No items._")
        else:
            lines.extend(
                [
                    "| # | Name | Quantity | Unit Price | Price |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for idx, item in enumerate(pred.items):
                lines.append(
                    f"| {idx} | {_md_escape_cell(item.name)} | "
                    f"{_fmt_opt_float(item.quantity)} | "
                    f"{_fmt_opt_float(item.unit_price)} | {item.price} |"
                )
        lines.extend(
            [
                "",
                "### Totals",
                "",
                "| Field | Amount |",
                "| --- | --- |",
                f"| `subtotal` | {pred.subtotal} |",
                f"| `tax` | {pred.tax} |",
                f"| `total` | {pred.total} |",
            ]
        )

    lines.extend(["", "## Raw Content", ""])
    if result.raw_content is None:
        lines.append("_None_")
    else:
        lines.append(_md_fenced_block(result.raw_content, "json"))

    return "\n".join(lines).rstrip() + "\n"


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
        json_path, md_path = timestamped_output_paths(args.output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(
            format_markdown_report(result),
            encoding="utf-8",
        )
        print(f"\nSaved full benchmark results to {json_path} and {md_path}")

    return 0 if result.error is None else 1
