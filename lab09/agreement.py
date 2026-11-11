
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    ms_list = list(ms)
    if not ms_list:
        raise AgreementError("No samples")
    
    for sample in ms_list:
        verified_sample = _finite(sample, "latency report")
        if verified_sample < 0:
            raise AgreementError("Negative latency")

    n = len(ms_list)
    mean = sum(ms_list) / n
    worst = max(ms_list)

    s = sorted(ms_list)
    p_list = []
    for q in LATENCY_PERCENTILES:
        p = _percentile(s, q)
        p_list.append(p)

    jitter = p_list[1] - p_list[0]

    return {
        "n": n,
        "mean": mean,
        "worst": worst,
        "p50": p_list[0],
        "p95": p_list[1],
        "p99": p_list[2],
        "jitter": jitter
    }



def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    validated_base = _finite(base_ms, "speedup")
    validated_other = _finite(other_ms, "speedup")
    if validated_base <= 0 or validated_other <= 0:
        raise AgreementError("nonpositive time")

    factor = validated_base / validated_other
    faster = factor > 1.0
    saved_ms = validated_base - validated_other

    return {
        "factor": factor,
        "faster": faster,
        "saved_ms": saved_ms
    }

def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    if len(a) != len(b):
        raise AgreementError("unequal lengths")

    if not a or not b:
        raise AgreementError("empty list")

    for sample_a, sample_b in zip(a, b):
        _finite(sample_a, "max_abs_dev")
        _finite(sample_b, "max_abs_dev")

    best_i = 0
    best = abs(a[0] - b[0])

    for i in range(1, len(a)):
        delta = abs(a[i] - b[i])
        if delta > best:
            best_i = i
            best = delta

    return {
        "max_abs": best,
        "index": best_i,
        "a": a[best_i],
        "b": b[best_i],
        "n": len(a)
    }


def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    if len(a) != len(b):
        raise AgreementError("unequal lengths")

    if not a or not b:
        raise AgreementError("empty list")

    for sample_a, sample_b in zip(a, b):
        _finite(sample_a, "max_abs_dev")
        _finite(sample_b, "max_abs_dev")

    if floor <= 0:
        raise AgreementError("nonpositive floor")

    floored = 0
    denom_min = float("inf")
    best = 0.0
    best_i = 0

    for i in range(len(a)):
        d = abs(b[i])
        denom_min = min(denom_min, d)
        if d < floor:
            floored += 1
        
        r = abs(a[i] - b[i]) / max(d, floor)
        if r > best:
            best = r
            best_i = i

    return{
        "max_rel": best,
        "index": best_i,
        "floored": floored,
        "denom_min": denom_min,
        "n": len(a)
    }


def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    if len(rows_a) != len(rows_b):
        raise AgreementError("unequal lengths")

    if not rows_a or not rows_b:
        raise AgreementError("empty list")

    for row_a, row_b in zip(rows_a, rows_b):
        if len(row_a) != len(row_b):
            raise AgreementError("unequal lengths")

        if not row_a or not row_b:
            raise AgreementError("empty list")
        
        for sample_a, sample_b in zip(row_a, row_b):
            _finite(sample_a, "top1_agreement")
            _finite(sample_b, "top1_agreement")

    agree = 0
    first_disagreement = None

    for k, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        best_a = float("-inf")
        i_a = 0
        best_b = float("-inf")
        i_b = 0
        for c, (sample_a, sample_b) in enumerate(zip(row_a, row_b)):

            if sample_a > best_a:
                best_a = sample_a
                i_a = c

            if sample_b > best_b:
                best_b = sample_b
                i_b = c

        if i_a == i_b:
            agree += 1
        elif first_disagreement is None:
            first_disagreement = k

    disagree = len(rows_a) - agree
    rate = agree / len(rows_a)
        
    return {
        "n": len(rows_a),
        "agree": agree,
        "disagree": disagree,
        "rate": rate,
        "first_disagreement": first_disagreement
    }


def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    if not rows:
        raise AgreementError("empty list")

    if tol < 0:
        raise AgreementError("negative tolerance")
    margins = []
    for row in rows:
        if len(row) < 2:
            raise AgreementError("row len is less than 2")

        for v in row:
            _finite(v, "decision_margin")

        sorted_row = sorted(row, reverse=True)
        margins.append(sorted_row[0] - sorted_row[1])

    minimum = float("inf")
    argmin = 0
    for i, margin in enumerate(margins):
        if margin < minimum:
            minimum = margin
            argmin = i

    median = _percentile(margins, 50)
    mean = sum(margins) / len(margins)
    below_tol = 0
    for margin in margins:
        if margin < tol:
            below_tol += 1
    safe = below_tol == 0

    return {
        "n": len(margins),
        "min": minimum,
        "median": median,
        "mean": mean,
        "argmin": argmin,
        "below_tol": below_tol,
        "safe": safe
    }


#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()