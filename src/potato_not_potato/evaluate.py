import math
from dataclasses import asdict, dataclass
from typing import Any

from potato_not_potato.models import Receipt, ReceiptItem

ITEM_FIELDS = ("name", "quantity", "unit_price", "price")
SUMMARY_FIELDS = ("subtotal", "tax", "total")


@dataclass(frozen=True, slots=True)
class FieldMismatch:
    location: str
    field: str
    expected: Any
    actual: Any


@dataclass(frozen=True, slots=True)
class AccuracyMetrics:
    overall_accuracy: float
    matched_fields: int
    total_fields: int
    exact_items: int
    expected_items: int
    predicted_items: int
    item_exact_rate: float
    field_accuracies: dict[str, float]
    mismatches: list[FieldMismatch]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _floats_equal(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6)


def _item_field_matches(pred: ReceiptItem, gt: ReceiptItem) -> dict[str, bool]:
    return {
        "name": pred.name == gt.name,
        "quantity": _floats_equal(pred.quantity, gt.quantity),
        "unit_price": _floats_equal(pred.unit_price, gt.unit_price),
        "price": _floats_equal(pred.price, gt.price),
    }


def _pair_score(pred: ReceiptItem, gt: ReceiptItem) -> int:
    matches = _item_field_matches(pred, gt)
    return sum(1 for matched in matches.values() if matched)


def _align_items(
    predicted: list[ReceiptItem],
    expected: list[ReceiptItem],
) -> list[tuple[int | None, int | None]]:
    """Order-preserving DP alignment maximizing matched fields across item pairs."""
    n = len(predicted)
    m = len(expected)

    # dp[i][j] = (matched_fields, exact_items) for suffixes predicted[i:] & expected[j:]
    dp: list[list[tuple[int, int]]] = [
        [(0, 0) for _ in range(m + 1)] for _ in range(n + 1)
    ]

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            score = _pair_score(predicted[i], expected[j])
            exact_bonus = 1 if score == len(ITEM_FIELDS) else 0
            pair_next = dp[i + 1][j + 1]
            pair_val = (pair_next[0] + score, pair_next[1] + exact_bonus)

            best = dp[i + 1][j]
            best = max(best, dp[i][j + 1])
            # Require at least 1 matching field to align two items rather than skipping
            if score > 0 and pair_val >= best:
                best = pair_val
            dp[i][j] = best

    alignment: list[tuple[int | None, int | None]] = []
    i, j = 0, 0
    while i < n and j < m:
        score = _pair_score(predicted[i], expected[j])
        exact_bonus = 1 if score == len(ITEM_FIELDS) else 0
        pair_next = dp[i + 1][j + 1]
        pair_val = (pair_next[0] + score, pair_next[1] + exact_bonus)

        if score > 0 and dp[i][j] == pair_val:
            alignment.append((i, j))
            i += 1
            j += 1
        elif dp[i][j] == dp[i][j + 1]:
            alignment.append((None, j))
            j += 1
        else:
            alignment.append((i, None))
            i += 1

    while i < n:
        alignment.append((i, None))
        i += 1
    while j < m:
        alignment.append((None, j))
        j += 1

    return alignment


def evaluate_receipt(predicted: Receipt, ground_truth: Receipt) -> AccuracyMetrics:
    alignment = _align_items(predicted.items, ground_truth.items)

    mismatches: list[FieldMismatch] = []
    matched_fields = 0
    exact_items = 0

    per_field_hits: dict[str, int] = {f: 0 for f in (*ITEM_FIELDS, *SUMMARY_FIELDS)}
    item_denom = max(len(ground_truth.items), len(predicted.items))

    for pred_idx, gt_idx in alignment:
        if pred_idx is not None and gt_idx is not None:
            pred_item = predicted.items[pred_idx]
            gt_item = ground_truth.items[gt_idx]
            field_matches = _item_field_matches(pred_item, gt_item)
            if all(field_matches.values()):
                exact_items += 1
            for field in ITEM_FIELDS:
                if field_matches[field]:
                    matched_fields += 1
                    per_field_hits[field] += 1
                else:
                    mismatches.append(
                        FieldMismatch(
                            location=f"items[{gt_idx}]",
                            field=field,
                            expected=getattr(gt_item, field),
                            actual=getattr(pred_item, field),
                        )
                    )
        elif gt_idx is not None:
            gt_item = ground_truth.items[gt_idx]
            mismatches.append(
                FieldMismatch(
                    location=f"items[{gt_idx}]",
                    field="<missing_item>",
                    expected=gt_item.model_dump(),
                    actual=None,
                )
            )
        elif pred_idx is not None:
            pred_item = predicted.items[pred_idx]
            mismatches.append(
                FieldMismatch(
                    location=f"predicted_items[{pred_idx}]",
                    field="<extra_item>",
                    expected=None,
                    actual=pred_item.model_dump(),
                )
            )

    for summary_field in SUMMARY_FIELDS:
        exp_val: float = getattr(ground_truth, summary_field)
        act_val: float = getattr(predicted, summary_field)
        if _floats_equal(act_val, exp_val):
            matched_fields += 1
            per_field_hits[summary_field] = 1
        else:
            mismatches.append(
                FieldMismatch(
                    location="receipt",
                    field=summary_field,
                    expected=exp_val,
                    actual=act_val,
                )
            )

    total_fields = item_denom * len(ITEM_FIELDS) + len(SUMMARY_FIELDS)
    overall_accuracy = matched_fields / total_fields if total_fields > 0 else 1.0
    item_exact_rate = exact_items / item_denom if item_denom > 0 else 1.0

    field_accuracies: dict[str, float] = {}
    for field in ITEM_FIELDS:
        field_accuracies[field] = (
            per_field_hits[field] / item_denom if item_denom > 0 else 1.0
        )
    for field in SUMMARY_FIELDS:
        field_accuracies[field] = float(per_field_hits[field])

    return AccuracyMetrics(
        overall_accuracy=overall_accuracy,
        matched_fields=matched_fields,
        total_fields=total_fields,
        exact_items=exact_items,
        expected_items=len(ground_truth.items),
        predicted_items=len(predicted.items),
        item_exact_rate=item_exact_rate,
        field_accuracies=field_accuracies,
        mismatches=mismatches,
    )
