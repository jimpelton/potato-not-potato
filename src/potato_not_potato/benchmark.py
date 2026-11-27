import time
from dataclasses import asdict, dataclass
from typing import Any

from openai import LengthFinishReasonError, OpenAI
from pydantic import ValidationError

from potato_not_potato.evaluate import AccuracyMetrics, evaluate_receipt
from potato_not_potato.models import Receipt

SYSTEM_PROMPT = """\
Extract structured receipt data from the provided raw receipt text.

Follow these rules:
1. Items (`items`):
   - Extract each purchased line item in order of appearance.
   - Strip leading UPC/PLU item codes from `name` and join wrapped item description lines with single spaces.
   - Populate `quantity` and `unit_price` ONLY when the receipt explicitly includes a weight or count breakdown line (e.g., `0.65 lb @ $15.99/lb` or `2 @ 3.00`). Otherwise set both `quantity` and `unit_price` to `null`.
   - Include discount or coupon lines in `items` with a negative `price` (and `quantity: null`, `unit_price: null`) ONLY if they have an associated dollar amount. Ignore blank `Discount:` lines with no amount.
   - Exclude receipt header lines (date, receipt number, cashier) and tender/payment lines (credit card, total tendered, change).
   - Clean obvious character corruption in text and numbers (such as `0Z`/`0%` for `OZ` or commas used as decimal points).
2. Totals:
   - `subtotal`: receipt subtotal before tax.
   - `tax`: state/sales tax amount.
   - `total`: final total charged.\
"""


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    wall_time_seconds: float
    prompt_tokens: int | None
    cached_prompt_tokens: int | None
    completion_tokens: int | None
    prompt_ms: float | None
    generation_ms: float | None
    prefill_tokens_per_sec: float | None
    generation_tokens_per_sec: float | None
    timings_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    model: str
    base_url: str
    performance: PerformanceMetrics
    accuracy: AccuracyMetrics | None
    predicted: Receipt | None
    raw_content: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "performance": self.performance.to_dict(),
            "accuracy": self.accuracy.to_dict() if self.accuracy else None,
            "predicted": self.predicted.model_dump() if self.predicted else None,
            "raw_content": self.raw_content,
            "error": self.error,
        }


def _as_float(val: Any) -> float | None:
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _as_int(val: Any) -> int | None:
    if isinstance(val, int):
        return val
    return None


def _extract_performance_metrics(
    response: Any,
    wall_time_seconds: float,
) -> PerformanceMetrics:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

    model_extra: dict[str, Any] = getattr(response, "model_extra", None) or {}
    timings = model_extra.get("timings")

    if isinstance(timings, dict):
        prompt_n = _as_int(timings.get("prompt_n"))
        cache_n = _as_int(timings.get("cache_n"))
        predicted_n = _as_int(timings.get("predicted_n"))
        prompt_ms = _as_float(timings.get("prompt_ms"))
        predicted_ms = _as_float(timings.get("predicted_ms"))
        prefill_tps = _as_float(timings.get("prompt_per_second"))
        gen_tps = _as_float(timings.get("predicted_per_second"))

        if prefill_tps is None and prompt_n and prompt_ms and prompt_ms > 0:
            prefill_tps = prompt_n / (prompt_ms / 1000.0)
        if gen_tps is None and predicted_n and predicted_ms and predicted_ms > 0:
            gen_tps = predicted_n / (predicted_ms / 1000.0)

        return PerformanceMetrics(
            wall_time_seconds=wall_time_seconds,
            prompt_tokens=prompt_n if prompt_n is not None else prompt_tokens,
            cached_prompt_tokens=cache_n,
            completion_tokens=(
                predicted_n if predicted_n is not None else completion_tokens
            ),
            prompt_ms=prompt_ms,
            generation_ms=predicted_ms,
            prefill_tokens_per_sec=prefill_tps,
            generation_tokens_per_sec=gen_tps,
            timings_source="llama.cpp",
        )

    # Fallback when server does not emit llama.cpp `timings`
    gen_tps_fallback = (
        completion_tokens / wall_time_seconds
        if completion_tokens and wall_time_seconds > 0
        else None
    )
    return PerformanceMetrics(
        wall_time_seconds=wall_time_seconds,
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=None,
        completion_tokens=completion_tokens,
        prompt_ms=None,
        generation_ms=None,
        prefill_tokens_per_sec=None,
        generation_tokens_per_sec=gen_tps_fallback,
        timings_source="wall_clock_fallback",
    )


def run_benchmark(
    *,
    receipt_text: str,
    ground_truth: Receipt,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    cache_prompt: bool = False,
) -> BenchmarkResult:
    client = OpenAI(base_url=base_url, api_key=api_key)

    start = time.perf_counter()
    try:
        response = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": receipt_text},
            ],
            response_format=Receipt,
            temperature=temperature,
            extra_body={"cache_prompt": cache_prompt},
        )
    except LengthFinishReasonError as exc:
        wall_time = time.perf_counter() - start
        performance = _extract_performance_metrics(exc.completion, wall_time)
        raw = (
            exc.completion.choices[0].message.content
            if exc.completion.choices
            else None
        )
        return BenchmarkResult(
            model=exc.completion.model or model,
            base_url=base_url,
            performance=performance,
            accuracy=None,
            predicted=None,
            raw_content=raw,
            error=str(exc),
        )
    except ValidationError as exc:
        wall_time = time.perf_counter() - start
        performance = PerformanceMetrics(
            wall_time_seconds=wall_time,
            prompt_tokens=None,
            cached_prompt_tokens=None,
            completion_tokens=None,
            prompt_ms=None,
            generation_ms=None,
            prefill_tokens_per_sec=None,
            generation_tokens_per_sec=None,
            timings_source="wall_clock_fallback",
        )
        return BenchmarkResult(
            model=model,
            base_url=base_url,
            performance=performance,
            accuracy=None,
            predicted=None,
            raw_content=None,
            error=f"Response failed schema validation: {exc}",
        )
    wall_time = time.perf_counter() - start

    performance = _extract_performance_metrics(response, wall_time)

    if not response.choices:
        return BenchmarkResult(
            model=response.model or model,
            base_url=base_url,
            performance=performance,
            accuracy=None,
            predicted=None,
            raw_content=None,
            error="Response contained no choices.",
        )

    message = response.choices[0].message
    raw_content = message.content
    predicted = message.parsed

    if predicted is None:
        refusal = getattr(message, "refusal", None)
        return BenchmarkResult(
            model=response.model or model,
            base_url=base_url,
            performance=performance,
            accuracy=None,
            predicted=None,
            raw_content=raw_content,
            error=f"Failed to parse structured Receipt response (refusal={refusal!r}).",
        )

    accuracy = evaluate_receipt(predicted, ground_truth)
    return BenchmarkResult(
        model=response.model or model,
        base_url=base_url,
        performance=performance,
        accuracy=accuracy,
        predicted=predicted,
        raw_content=raw_content,
    )
