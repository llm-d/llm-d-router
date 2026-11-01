import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import trend


TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "targets", "optimized-baseline-job1.yaml")
CONFIG = trend.load_trend_config(TARGET)
DIGEST = "digest-a"


def run_record(day, status="SUCCESS", epp_peak_cpu=2000, p95=9.0, envoy_peak_cpu=2500, commit=None, target_id=CONFIG.target_id, digest=DIGEST):
    return {
        "schema_version": trend.RUN_RECORD_SCHEMA_VERSION,
        "test_name": target_id,
        "started_at": f"2026-10-{day:02d}T09:00:00Z",
        "status": status,
        "target": {"id": target_id, "digest": digest, "setup": {}},
        "provenance": {"epp_commit": commit or f"commit{day:02d}", "images": [], "image_ids": {}},
        "containers": {
            "epp": {"idle_cpu_m": 200, "idle_mem_mib": 25, "peak_cpu_m": epp_peak_cpu, "peak_mem_mib": 70},
            "envoy-proxy": {"idle_cpu_m": 30, "idle_mem_mib": 17, "peak_cpu_m": envoy_peak_cpu, "peak_mem_mib": 50},
            "TOTAL": {"idle_cpu_m": 230, "idle_mem_mib": 42, "peak_cpu_m": epp_peak_cpu + envoy_peak_cpu, "peak_mem_mib": 120},
        },
        "scheduler_latency_ms": {"p50": 3.5, "p95": p95, "p99": 16.0, "mean": 4.0},
    }


def history(previous=None, tonight=None):
    baseline = [run_record(day) for day in range(9, 16)]
    return baseline + [run_record(16, **(previous or {})), run_record(17, **(tonight or {}))]


def lasting(**values):
    return history(previous=values, tonight=values)


def result_for(verdict, name):
    return next(result for result in verdict["metrics"] if result["name"] == name)


def with_metrics(*metrics):
    return CONFIG._replace(metrics=list(metrics))


class TrendConfigTests(unittest.TestCase):
    def test_nightly_target_config(self):
        self.assertEqual(CONFIG.target_id, "optimized-baseline-job1")
        self.assertEqual((CONFIG.window_days, CONFIG.min_baseline_runs), (7, 4))
        self.assertEqual(len(CONFIG.metrics), 16)
        self.assertEqual(sum(metric.alerts for metric in CONFIG.metrics), 7)
        self.assertEqual(CONFIG.metrics[1].path, ("containers", "epp", "peak_cpu_m"))
        metrics = {metric.name: metric for metric in CONFIG.metrics}
        self.assertFalse(metrics["epp idle CPU (m)"].alerts)
        for statistic in ("p50", "p95", "p99", "mean"):
            self.assertEqual(metrics[f"scheduler {statistic} latency (ms)"].threshold, 0.25)

    def test_metric_defaults_and_validation(self):
        metric = trend.parse_metric({"name": "m", "path": "a.b"})
        self.assertEqual((metric.direction, metric.threshold, metric.alerts), ("lower", 0.10, True))
        invalid = {
            "direction": {"direction": "sideways"},
            "unknown key": {"max_threshold": 0.25},
            "quoted alerts": {"alerts": "false"},
        }
        for case, fields in invalid.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    trend.parse_metric({"name": "m", "path": "a.b", **fields})

    def load_config(self, trend_section):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "bench.yaml"
            target.write_text(f"setup: {{}}\ntrend: {trend_section}\n")
            return trend.load_trend_config(str(target))

    def test_invalid_trend_sections_are_rejected(self):
        metric = "{name: m, path: a.b}"
        invalid = {
            "empty metrics": "{metrics: []}",
            "misspelled key": f"{{window_day: 14, metrics: [{metric}]}}",
        }
        for case, section in invalid.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    self.load_config(section)
        self.assertEqual(self.load_config(f"{{window_days: 14, metrics: [{metric}]}}").window_days, 14)


class BaselineSelectionTests(unittest.TestCase):
    def test_window_is_the_seven_days_before_the_run(self):
        records = [run_record(day) for day in range(5, 18)]
        baseline = trend.select_baseline(records, records[-1], 7)
        self.assertEqual([record["started_at"][:10] for record in baseline], [f"2026-10-{day}" for day in range(10, 17)])

    def test_failed_runs_and_other_targets_or_digests_are_excluded(self):
        records = [
            run_record(14, status="FAILED"),
            run_record(15, digest="digest-b"),
            run_record(16, target_id="other-target"),
            run_record(16),
            run_record(17),
        ]
        self.assertEqual(trend.select_baseline(records, records[-1], 7), [records[3]])


class EvaluateTests(unittest.TestCase):
    def test_no_runs(self):
        self.assertEqual(trend.evaluate([], CONFIG)["status"], "no_runs")

    def test_failed_tonight_is_not_compared(self):
        verdict = trend.evaluate(history(tonight={"status": "FAILED"}), CONFIG)
        self.assertEqual(verdict["status"], "run_failed")
        self.assertEqual(verdict["metrics"], [])

    def test_fewer_than_four_baseline_runs_is_insufficient(self):
        verdict = trend.evaluate([run_record(day) for day in range(13, 18)], CONFIG)
        self.assertEqual(verdict["status"], "insufficient_baseline")
        self.assertEqual(len(verdict["baseline_runs"]), 3)
        self.assertEqual(trend.evaluate([run_record(day) for day in range(12, 18)], CONFIG)["status"], "ok")

    def test_no_previous_run_is_insufficient(self):
        self.assertEqual(trend.evaluate([run_record(17)], CONFIG)["status"], "insufficient_baseline")

    def test_new_digest_starts_a_new_baseline(self):
        records = [run_record(day) for day in range(9, 17)] + [run_record(17, digest="digest-b")]
        self.assertEqual(trend.evaluate(records, CONFIG)["status"], "insufficient_baseline")

    def test_baseline_is_the_week_before_the_previous_run(self):
        verdict = trend.evaluate(history(), CONFIG)
        self.assertEqual([run[:10] for run in verdict["baseline_runs"]], [f"2026-10-{day:02d}" for day in range(9, 16)])

    def test_lasting_step_regresses(self):
        verdict = trend.evaluate(history(previous={"epp_peak_cpu": 2250}, tonight={"epp_peak_cpu": 2201}), CONFIG)
        self.assertEqual(verdict["status"], "regressed")
        result = result_for(verdict, "epp peak CPU (m)")
        self.assertEqual(result["verdict"], "regressed")
        self.assertEqual((result["tonight"], result["previous"]), (2201, 2250))
        self.assertEqual(result["baseline"], 2000)
        self.assertAlmostEqual(result["delta"], 0.1005)
        self.assertEqual(result["runs"], 7)

    def test_lasting_step_is_reported_once(self):
        records = [run_record(day) for day in range(8, 15)]
        records += [run_record(day, epp_peak_cpu=2500) for day in range(15, 18)]
        self.assertEqual(trend.evaluate(records[:-1], CONFIG)["status"], "regressed")
        self.assertEqual(trend.evaluate(records, CONFIG)["status"], "ok")

    def test_one_bad_run_is_reported_without_regressing(self):
        verdict = trend.evaluate(history(tonight={"epp_peak_cpu": 2500}), CONFIG)
        self.assertEqual(verdict["status"], "ok")
        self.assertEqual(result_for(verdict, "epp peak CPU (m)")["verdict"], "over threshold, first run")
        recovered = trend.evaluate(history(previous={"epp_peak_cpu": 2500}), CONFIG)
        self.assertEqual(recovered["status"], "ok")
        self.assertEqual(result_for(recovered, "epp peak CPU (m)")["verdict"], "within threshold")

    def test_exactly_ten_percent_and_improvements_do_not_regress(self):
        for value in (2200, 1500):
            with self.subTest(value=value):
                verdict = trend.evaluate(lasting(epp_peak_cpu=value), CONFIG)
                self.assertEqual(verdict["status"], "ok")
                self.assertEqual(result_for(verdict, "epp peak CPU (m)")["verdict"], "within threshold")

    def test_inside_the_baseline_range_does_not_regress(self):
        records = lasting(epp_peak_cpu=2250)
        records[3]["containers"]["epp"]["peak_cpu_m"] = 2300
        verdict = trend.evaluate(records, CONFIG)
        self.assertEqual(verdict["status"], "ok")
        self.assertEqual(result_for(verdict, "epp peak CPU (m)")["verdict"], "within threshold")

    def test_per_metric_threshold(self):
        loose = trend.Metric("epp peak CPU (m)", ("containers", "epp", "peak_cpu_m"), "lower", 0.30, True)
        config = with_metrics(loose)
        self.assertEqual(trend.evaluate(lasting(epp_peak_cpu=2500), config)["status"], "ok")
        self.assertEqual(trend.evaluate(lasting(epp_peak_cpu=2700), config)["status"], "regressed")

    def test_higher_is_better_metric_regresses_when_it_drops(self):
        throughput = trend.Metric("throughput", ("client", "throughput"), "higher", 0.10, True)
        records = history()
        for record in records:
            record["client"] = {"throughput": 100}
        for record in records[-2:]:
            record["client"]["throughput"] = 80
        verdict = trend.evaluate(records, with_metrics(throughput))
        self.assertEqual(verdict["status"], "regressed")
        for record in records[-2:]:
            record["client"]["throughput"] = 130
        self.assertEqual(trend.evaluate(records, with_metrics(throughput))["status"], "ok")

    def test_failed_run_between_does_not_break_the_pair(self):
        records = [run_record(day) for day in range(8, 15)]
        records += [run_record(15, epp_peak_cpu=2500), run_record(16, status="FAILED"), run_record(17, epp_peak_cpu=2500)]
        verdict = trend.evaluate(records, CONFIG)
        self.assertEqual(verdict["status"], "regressed")
        self.assertEqual(verdict["range_base_commit"], "commit14")

    def test_scheduler_latency_regresses(self):
        verdict = trend.evaluate(lasting(p95=18.5), CONFIG)
        self.assertEqual(verdict["status"], "regressed")
        self.assertEqual(result_for(verdict, "scheduler p95 latency (ms)")["verdict"], "regressed")

    def test_report_only_metrics_never_regress(self):
        verdict = trend.evaluate(lasting(envoy_peak_cpu=5000), CONFIG)
        self.assertEqual(verdict["status"], "ok")
        self.assertEqual(result_for(verdict, "envoy-proxy peak CPU (m)")["verdict"], "over threshold, report only")
        self.assertEqual(result_for(verdict, "TOTAL peak CPU (m)")["verdict"], "over threshold, report only")

    def test_unmeasured_values_are_missing_not_zero(self):
        records = history(tonight={"p95": 0.0})
        for record in records[:4]:
            record["scheduler_latency_ms"]["p99"] = 0.0
        verdict = trend.evaluate(records, CONFIG)
        self.assertEqual(result_for(verdict, "scheduler p95 latency (ms)")["verdict"], "no data")
        self.assertEqual(result_for(verdict, "scheduler p99 latency (ms)")["verdict"], "insufficient baseline")
        self.assertEqual(verdict["status"], "ok")

    def test_commit_range_starts_at_the_last_run_before_the_previous_one(self):
        verdict = trend.evaluate(history(), CONFIG)
        self.assertEqual(verdict["range_base_commit"], "commit15")
        self.assertEqual(verdict["tonight"]["epp_commit"], "commit17")


class IssueBodyTests(unittest.TestCase):
    def test_body_has_marker_regressed_metrics_and_compare_link(self):
        verdict = trend.evaluate(lasting(epp_peak_cpu=2500), CONFIG)
        body = trend.render_issue_body(verdict, "llm-d/llm-d-router", "https://example.com/run/1")
        self.assertTrue(body.startswith("<!-- perf-trend:optimized-baseline-job1 -->\n"))
        self.assertIn("| Previous run |", body)
        self.assertIn("| epp peak CPU (m) | 2500.00 | 2500.00 | 2000.00 |", body)
        self.assertNotIn("epp idle CPU (m)", body)
        self.assertIn("https://github.com/llm-d/llm-d-router/compare/commit15...commit17", body)
        self.assertIn("https://example.com/run/1", body)

    def test_missing_commit_has_no_compare_link(self):
        records = lasting(epp_peak_cpu=2500)
        records[-1]["provenance"]["epp_commit"] = None
        verdict = trend.evaluate(records, CONFIG)
        body = trend.render_issue_body(verdict, "llm-d/llm-d-router")
        self.assertNotIn("/compare/", body)
        self.assertIn("commit range is unavailable", body)


class FileTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.results = self.directory / "results"
        self.output = self.directory / "out"
        self.runs = self.results / "runs" / CONFIG.target_id
        self.runs.mkdir(parents=True)
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("GITHUB_STEP_SUMMARY", None)

    def write_records(self, records):
        for index, record in enumerate(records):
            (self.runs / f"{index:02d}.json").write_text(json.dumps(record))

    def run_main(self):
        with mock.patch("builtins.print"):
            return trend.main([
                "--results-dir", str(self.results),
                "--target", TARGET,
                "--output-dir", str(self.output),
            ])

    def test_records_load_in_time_order(self):
        records = history()
        self.write_records(list(reversed(records)))
        loaded = trend.load_records(str(self.results), CONFIG.target_id)
        self.assertEqual([record["started_at"] for record in loaded], [record["started_at"] for record in records])

    def test_missing_runs_folder_is_no_runs(self):
        self.assertEqual(trend.load_records(str(self.directory / "empty"), CONFIG.target_id), [])

    def test_regression_exits_zero_and_writes_issue_body(self):
        self.write_records(lasting(epp_peak_cpu=2500))
        summary_file = self.directory / "step-summary.md"
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary_file)
        self.assertEqual(self.run_main(), 0)
        verdict = json.loads((self.output / "trend-verdict.json").read_text())
        self.assertEqual(verdict["status"], "regressed")
        self.assertTrue((self.output / "trend-issue.md").exists())
        self.assertIn("Perf trend: optimized-baseline-job1", summary_file.read_text())

    def test_no_regression_writes_no_issue_body(self):
        self.write_records(history())
        self.assertEqual(self.run_main(), 0)
        self.assertFalse((self.output / "trend-issue.md").exists())
        self.assertIn("| epp peak CPU (m) |", (self.output / "trend-summary.md").read_text())

    def test_records_from_another_schema_version_are_skipped(self):
        records = lasting(epp_peak_cpu=2500)
        self.write_records(records)
        old = {key: value for key, value in run_record(18).items() if key != "target"}
        (self.runs / "old.json").write_text(json.dumps({**old, "schema_version": 1, "setup": {}}))
        loaded = trend.load_records(str(self.results), CONFIG.target_id)
        self.assertEqual(len(loaded), len(records))
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(json.loads((self.output / "trend-verdict.json").read_text())["status"], "regressed")

    def test_broken_records_raise(self):
        cases = {
            "unparsable": "{not json",
            "missing target": json.dumps({key: value for key, value in run_record(17).items() if key != "target"}),
        }
        for name, content in cases.items():
            with self.subTest(case=name):
                (self.runs / "bad.json").write_text(content)
                with self.assertRaises(ValueError):
                    self.run_main()


if __name__ == "__main__":
    unittest.main()
