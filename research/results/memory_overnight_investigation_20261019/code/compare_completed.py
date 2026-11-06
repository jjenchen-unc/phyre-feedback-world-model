"""Descriptive paired analysis; never pool incomplete arms or claim significance."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent
SOURCE = RESULTS / "q0_first_feedback_20261018"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(candidate, reference):
    assert set(candidate) == set(reference)
    tasks = sorted(candidate)
    by_budget = {}
    for budget in range(1, 10):
        a = {t: candidate[t] is not None and candidate[t] <= budget for t in tasks}
        b = {t: reference[t] is not None and reference[t] <= budget for t in tasks}
        gains = [t for t in tasks if a[t] and not b[t]]
        losses = [t for t in tasks if b[t] and not a[t]]
        templates = {}
        for template in sorted({t.split(":")[0] for t in tasks}):
            subset = [t for t in tasks if t.split(":")[0] == template]
            templates[template] = {
                "n": len(subset),
                "candidate_solved": sum(a[t] for t in subset),
                "reference_solved": sum(b[t] for t in subset),
                "net_gain": sum(int(a[t]) - int(b[t]) for t in subset),
            }
        net = len(gains) - len(losses)
        leave_one_out = {
            t: (net - v["net_gain"]) / (len(tasks) - v["n"])
            for t, v in templates.items()
        }
        by_budget[str(budget)] = {
            "n": len(tasks), "candidate_solved": sum(a.values()),
            "reference_solved": sum(b.values()), "gains": gains, "losses": losses,
            "net_percentage_points": 100 * net / len(tasks),
            "templates": templates,
            "leave_one_template_out_delta_fraction": leave_one_out,
        }
    return by_budget


def coverage(panel, tasks):
    result = {}
    for task in tasks:
        path = SOURCE / "evaluation" / panel / "residual" / task.replace(":", "_") / "scores.npz"
        with np.load(path) as z:
            result[task] = int(z["labels"].sum())
    return result


def run():
    out = {
        "interpretation": "Descriptive only; template deletion is a sensitivity check, not a confidence interval. Multiple interventions on one evaluation panel do not establish a validated winner.",
        "experiments": {},
    }
    specifications = [
        ("closed_loop_interventions", RESULTS / "memory_overnight_policy_20261019" / "SUMMARY.json", ["q0", "residual"]),
        ("fresh_query_balance", RESULTS / "query_balance_training_20261019" / "SUMMARY_128.json", ["uniform", "original_q0"]),
        ("centering_checkpoint_stability", RESULTS / "query_balance_centered_eval_20261019" / "SUMMARY.json", ["uniform", "balanced"]),
    ]
    for name, path, references in specifications:
        if not path.exists():
            out["experiments"][name] = {"status": "pending_complete_summary"}
            continue
        summary = read(path)
        assert summary["complete"]
        panels = {}
        for panel, data in summary["panels"].items():
            methods = data["methods"]
            labels = coverage(panel, methods[references[0]]["first_success"])
            comparisons = {}
            for method, values in methods.items():
                first = values["first_success"]
                assert set(first) == set(labels)
                assert all(labels[t] > 0 for t, k in first.items() if k is not None)
                comparisons[method] = {
                    "successes_among_covered_tasks": {
                        "solved_s9": sum(k is not None for k in first.values()),
                        "covered_n": sum(n > 0 for n in labels.values()),
                    },
                    "paired": {ref: compare(first, methods[ref]["first_success"]) for ref in references if ref != method},
                }
            panels[panel] = {"candidate_positive_counts": labels, "methods": comparisons}
        out["experiments"][name] = {"status": "complete", "summary_sha256": digest(path), "panels": panels}
    fresh = RESULTS / "query_balance_training_20261019" / "SUMMARY_128.json"
    centered = RESULTS / "query_balance_centered_eval_20261019" / "SUMMARY.json"
    if fresh.exists() and centered.exists():
        raw_data, centered_data = read(fresh), read(centered)
        assert raw_data['complete'] and centered_data['complete']
        factorial = {}
        for panel in ['cross', 'within']:
            methods = centered_data['panels'][panel]['methods']
            for arm in ['uniform', 'balanced']:
                assert methods[arm]['first_success'] == raw_data['panels'][panel]['methods'][arm]['first_success']
            counts = {m: v['counts']['s9'] for m, v in methods.items()}
            assert len({v['n'] for v in methods.values()}) == 1
            factorial[panel] = {
                'n': methods['uniform']['n'], 's9_counts': counts,
                'balance_delta_raw_count': counts['balanced'] - counts['uniform'],
                'balance_delta_centered_count': counts['balanced_centered'] - counts['uniform_centered'],
                'centering_delta_uniform_count': counts['uniform_centered'] - counts['uniform'],
                'centering_delta_balanced_count': counts['balanced_centered'] - counts['balanced'],
                'interaction_count': counts['balanced_centered'] - counts['balanced'] - counts['uniform_centered'] + counts['uniform'],
                'interpretation': 'Descriptive 2x2 comparison on same100scenes; no statistical significance or independent replication implied.',
            }
        out['fresh_2x2'] = factorial
    (ROOT / "PAIRED_ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
    print({name: value["status"] for name, value in out["experiments"].items()})


if __name__ == "__main__":
    run()
