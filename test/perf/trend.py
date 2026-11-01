#!/usr/bin/env python3

import argparse
import json
import os
import statistics
import sys
from collections import namedtuple
from datetime import datetime, timedelta

from run_nightly_perf import RUN_RECORD_SCHEMA_VERSION, load_target, target_id

DEFAULT_THRESHOLD = 0.10
DEFAULT_WINDOW_DAYS = 7
DEFAULT_MIN_BASELINE_RUNS = 4
DEFAULT_REPOSITORY = "llm-d/llm-d-router"
DIRECTIONS = ("lower", "higher")
TREND_KEYS = ("window_days", "min_baseline_runs", "metrics")
METRIC_KEYS = ("name", "path", "direction", "threshold", "alerts")

Metric = namedtuple("Metric", ["name", "path", "direction", "threshold", "alerts"])
TrendConfig = namedtuple("TrendConfig", ["target_id", "window_days", "min_baseline_runs", "metrics"])


def check_keys(where, entry, allowed):
    # A misspelled key would otherwise fall back to its default without notice.
    unknown = set(entry) - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")


def parse_metric(entry):
    where = f"metric {entry.get('name')!r}"
    check_keys(where, entry, METRIC_KEYS)
    direction = entry.get("direction", "lower")
    if direction not in DIRECTIONS:
        raise ValueError(f"{where}: direction must be one of {DIRECTIONS}, got {direction!r}")
    alerts = entry.get("alerts", True)
    if not isinstance(alerts, bool):
        raise ValueError(f"{where}: alerts must be true or false, got {alerts!r}")
    return Metric(
        name=entry["name"],
        path=tuple(entry["path"].split(".")),
        direction=direction,
        threshold=float(entry.get("threshold", DEFAULT_THRESHOLD)),
        alerts=alerts,
    )


def load_trend_config(target_path):
    trend = load_target(target_path).get("trend") or {}
    check_keys(f"{target_path}: trend", trend, TREND_KEYS)
    metrics = [parse_metric(entry) for entry in trend.get("metrics") or []]
    if not metrics:
        raise ValueError(f"{target_path}: trend.metrics is empty")
    return TrendConfig(
        target_id=target_id(target_path),
        window_days=int(trend.get("window_days", DEFAULT_WINDOW_DAYS)),
        min_baseline_runs=int(trend.get("min_baseline_runs", DEFAULT_MIN_BASELINE_RUNS)),
        metrics=metrics,
    )


def issue_marker(test_name):
    return f"<!-- perf-trend:{test_name} -->"


def parse_time(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def load_records(results_dir, test_name):
    runs_dir = os.path.join(results_dir, "runs", test_name)
    if not os.path.isdir(runs_dir):
        return []
    records = []
    for name in os.listdir(runs_dir):
        if name.endswith(".json"):
            record = load_record(os.path.join(runs_dir, name))
            if record is not None:
                records.append(record)
    return sorted(records, key=lambda record: parse_time(record["started_at"]))


def load_record(path):
    with open(path) as f:
        record = json.load(f)
    # A record written by another schema version cannot be compared, so it is skipped rather than failing the check.
    if record.get("schema_version") != RUN_RECORD_SCHEMA_VERSION:
        print(f"Skipping {path}: run record schema_version {record.get('schema_version')!r}, expected {RUN_RECORD_SCHEMA_VERSION}")
        return None
    for key in ("started_at", "status", "target"):
        if key not in record:
            raise ValueError(f"{path}: run record is missing {key!r}")
    return record


def comparable(first, second):
    return all(first["target"].get(key) == second["target"].get(key) for key in ("id", "digest"))


def select_baseline(records, tonight, window_days):
    end = parse_time(tonight["started_at"])
    start = end - timedelta(days=window_days)
    return [
        record
        for record in records
        if record is not tonight
        and record["status"] == "SUCCESS"
        and comparable(record, tonight)
        and start <= parse_time(record["started_at"]) < end
    ]


def metric_value(record, metric):
    value = record
    for part in metric.path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    # The runner reports 0 for a metric it could not measure.
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def is_over(metric, tonight, baseline, values):
    # Beyond every baseline run as well as past the threshold, so a metric with a wide normal range does not fire on ordinary noise.
    if metric.direction == "lower":
        return (tonight - baseline) / baseline > metric.threshold and tonight > max(values)
    return (baseline - tonight) / baseline > metric.threshold and tonight < min(values)


def compare_metric(metric, baseline_records, candidate, min_runs):
    result = {"name": metric.name, "direction": metric.direction, "alerts": metric.alerts, "threshold": metric.threshold, "over": False}
    tonight = metric_value(candidate, metric)
    values = [value for value in (metric_value(record, metric) for record in baseline_records) if value is not None]
    result.update(tonight=tonight, previous=None, runs=len(values), baseline=None, delta=None, min=None, max=None)
    if tonight is None:
        result["verdict"] = "no data"
        return result
    if len(values) < min_runs:
        result["verdict"] = "insufficient baseline"
        return result
    baseline = statistics.median(values)
    over = is_over(metric, tonight, baseline, values)
    result.update(baseline=baseline, delta=(tonight - baseline) / baseline, min=min(values), max=max(values), over=over)
    if not over:
        result["verdict"] = "within threshold"
    elif metric.alerts:
        result["verdict"] = "over threshold"
    else:
        result["verdict"] = "over threshold, report only"
    return result


def compare(baseline_records, candidate, metrics, min_runs):
    return [compare_metric(metric, baseline_records, candidate, min_runs) for metric in metrics]


def previous_run(records, run):
    earlier = [
        record
        for record in records
        if record["status"] == "SUCCESS"
        and comparable(record, run)
        and parse_time(record["started_at"]) < parse_time(run["started_at"])
    ]
    return earlier[-1] if earlier else None


def apply_previous_run(results, previous_results):
    # A metric regresses only when the previous comparable run was over its threshold too.
    previous = {result["name"]: result for result in previous_results}
    for result in results:
        result["previous"] = previous[result["name"]]["tonight"]
        if result["verdict"] == "over threshold":
            result["verdict"] = "regressed" if previous[result["name"]]["over"] else "over threshold, first run"


def commit_of(record):
    return record.get("provenance", {}).get("epp_commit") or None


def evaluate(records, config):
    verdict = {"test_name": config.target_id, "window_days": config.window_days, "min_baseline_runs": config.min_baseline_runs, "metrics": []}
    if not records:
        verdict["status"] = "no_runs"
        return verdict
    tonight = records[-1]
    verdict["tonight"] = {"started_at": tonight["started_at"], "status": tonight["status"], "epp_commit": commit_of(tonight)}
    if tonight["status"] != "SUCCESS":
        verdict["status"] = "run_failed"
        return verdict
    previous = previous_run(records, tonight)
    # Tonight and the previous run share one baseline from before the previous run, so a lasting step counts on both.
    baseline = select_baseline(records, previous, config.window_days) if previous else []
    verdict["baseline_runs"] = [record["started_at"] for record in baseline]
    verdict["range_base_commit"] = commit_of(baseline[-1]) if baseline else None
    if len(baseline) < config.min_baseline_runs:
        verdict["status"] = "insufficient_baseline"
        return verdict
    verdict["metrics"] = compare(baseline, tonight, config.metrics, config.min_baseline_runs)
    apply_previous_run(verdict["metrics"], compare(baseline, previous, config.metrics, config.min_baseline_runs))
    regressed = any(result["verdict"] == "regressed" for result in verdict["metrics"])
    verdict["status"] = "regressed" if regressed else "ok"
    return verdict


def format_number(value):
    return "-" if value is None else f"{value:.2f}"


def format_delta(value):
    return "-" if value is None else f"{value * 100:+.1f}%"


def metrics_table(results):
    lines = [
        "| Metric | Tonight | Previous run | Baseline (median) | Delta | Threshold | Baseline range | Runs | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        spread = "-" if result["min"] is None else f"{result['min']:.2f}-{result['max']:.2f}"
        threshold = f"{result['threshold'] * 100:.0f}% {result['direction']} is better"
        lines.append(
            f"| {result['name']} | {format_number(result['tonight'])} | {format_number(result['previous'])} | {format_number(result['baseline'])} | "
            f"{format_delta(result['delta'])} | {threshold} | {spread} | {result['runs']} | {result['verdict']} |"
        )
    return "\n".join(lines)


RULE_TEXT = (
    "A metric regresses when tonight's run and the previous comparable run are both worse than the median of the "
    "successful runs from the {window_days} days before the previous run by more than its threshold, and worse than "
    "every one of them. Runs are comparable when they share the benchmark target and the digest of its setup, "
    "workload and simulator files. Router config edits keep the baseline."
)

STATUS_TEXT = {
    "no_runs": "No run records found.",
    "run_failed": "Tonight's run did not succeed, so nothing was compared.",
    "insufficient_baseline": "Fewer than {min_baseline_runs} comparable successful runs in the {window_days} days before the previous run, so nothing was compared.",
    "ok": "No alerting metric regressed.",
    "regressed": "At least one alerting metric regressed.",
}


def status_line(verdict):
    return STATUS_TEXT[verdict["status"]].format(
        min_baseline_runs=verdict["min_baseline_runs"],
        window_days=verdict["window_days"],
    )


def rule_line(verdict):
    return RULE_TEXT.format(window_days=verdict["window_days"])


def render_summary(verdict):
    lines = [f"## Perf trend: {verdict['test_name']}", "", status_line(verdict), "", rule_line(verdict), ""]
    if "tonight" in verdict:
        lines.append(f"Tonight: {verdict['tonight']['started_at']} (EPP commit `{verdict['tonight']['epp_commit'] or 'unknown'}`)")
    if "baseline_runs" in verdict:
        lines.append(f"Baseline: {len(verdict['baseline_runs'])} comparable successful runs")
    if verdict["metrics"]:
        lines += ["", metrics_table(verdict["metrics"])]
    return "\n".join(lines) + "\n"


def compare_link(repository, base, head):
    if not base or not head or base == head:
        return None
    return f"https://github.com/{repository}/compare/{base}...{head}"


def render_issue_body(verdict, repository, run_url=None):
    regressed = [result for result in verdict["metrics"] if result["verdict"] == "regressed"]
    lines = [
        issue_marker(verdict["test_name"]),
        f"The nightly perf run on {verdict['tonight']['started_at']} confirmed a regression in these metrics. "
        + rule_line(verdict),
        "",
        metrics_table(regressed),
        "",
    ]
    link = compare_link(repository, verdict.get("range_base_commit"), verdict["tonight"]["epp_commit"])
    if link:
        lines.append(f"Changes since the last run before the regression: {link}")
    else:
        lines.append("The commit range is unavailable: a run did not record its EPP commit, or both runs used the same commit.")
    if run_url:
        lines.append(f"Workflow run: {run_url}")
    return "\n".join(lines) + "\n"


def write_outputs(verdict, output_dir, repository, run_url):
    os.makedirs(output_dir, exist_ok=True)
    summary = render_summary(verdict)
    with open(os.path.join(output_dir, "trend-summary.md"), "w") as f:
        f.write(summary)
    with open(os.path.join(output_dir, "trend-verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, sort_keys=True)
        f.write("\n")
    if verdict["status"] == "regressed":
        with open(os.path.join(output_dir, "trend-issue.md"), "w") as f:
            f.write(render_issue_body(verdict, repository, run_url))
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(summary)
    return summary


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Compare the newest nightly perf run of a benchmark target with its trailing median")
    parser.add_argument("--results-dir", required=True, help="Results folder containing runs/<target-id>/*.json")
    parser.add_argument("--target", required=True, help="Benchmark target manifest; its trend section sets the compared metrics")
    parser.add_argument("--output-dir", required=True, help="Folder for trend-summary.md, trend-verdict.json and trend-issue.md")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY), help="owner/name used for compare links")
    parser.add_argument("--run-url", default=None, help="Workflow run URL to link from the issue")
    return parser.parse_args(argv)


def main(argv=None):
    # A regression exits 0; an error in this script propagates and fails the step.
    args = parse_args(argv)
    config = load_trend_config(args.target)
    records = load_records(args.results_dir, config.target_id)
    verdict = evaluate(records, config)
    print(write_outputs(verdict, args.output_dir, args.repository, args.run_url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
