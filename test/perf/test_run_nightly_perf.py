from contextlib import ExitStack
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

import run_nightly_perf as perf


LEGACY_HEADER = (
    "| Timestamp | Namespace | Router Config | Perf Job | Machine Family | "
    "Sim Replicas | EPP Images | Container | Idle CPU (m) | Idle Mem (MiB) | "
    "Peak CPU (m) | Peak Mem (MiB) | P50 Latency (ms) | P95 Latency (ms) | "
    "CPU Profile | Memory Profile | Status |\n"
)


def scheduler_metrics():
    before = {
        "count": 100,
        "buckets": {"+Inf": 100, "0.1": 100, "0.001": 50, "0.01": 90},
    }
    after = {
        "count": 200,
        "buckets": {"+Inf": 200, "0.1": 200, "0.001": 100, "0.01": 180},
    }
    return before, after


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.directory = Path(
            self.contexts.enter_context(tempfile.TemporaryDirectory())
        )
        self.contexts.enter_context(mock.patch.object(perf, "print", create=True))
        self.contexts.enter_context(
            mock.patch.object(
                perf.subprocess,
                "run",
                side_effect=AssertionError("Unexpected subprocess"),
            )
        )
        self.contexts.enter_context(
            mock.patch.object(
                perf.subprocess,
                "Popen",
                side_effect=AssertionError("Unexpected subprocess"),
            )
        )


class DeployEPPTests(OfflineTestCase):
    def deploy(self, config, **kwargs):
        config_path = self.directory / "router.yaml"
        config_text = yaml.safe_dump(config)
        config_path.write_text(config_text)
        output_path = self.directory / "overrides.yaml"
        namespace = "offline-tracing-preflight"
        expected_path = f"/tmp/test-overrides-{namespace}.yaml"
        real_open = open

        def local_open(path, *args, **open_kwargs):
            if os.fspath(path) == expected_path:
                path = output_path
            return real_open(path, *args, **open_kwargs)

        with (
            mock.patch.object(perf, "open", local_open, create=True),
            mock.patch.object(perf, "run_cmd") as run_cmd,
        ):
            perf.deploy_epp(
                namespace,
                "oci://example.invalid/router",
                "v0",
                str(config_path),
                **kwargs,
            )
        self.assertEqual(run_cmd.call_count, 2)
        helm_command = run_cmd.call_args_list[0].args[0]
        self.assertEqual(
            helm_command[4:8], ["-f", str(config_path), "-f", expected_path]
        )
        self.assertEqual(config_path.read_text(), config_text)
        return yaml.safe_load(output_path.read_text())

    def test_explicit_tracing_enabled_for_sampling_ratios(self):
        for ratio in ("0.01", "0.1", "1.0"):
            with self.subTest(ratio=ratio):
                config = {
                    "router": {
                        "epp": {"flags": {"tracing": "true"}},
                        "tracing": {"sampling": {"samplerArg": ratio}},
                    }
                }
                overrides = self.deploy(config)
                self.assertEqual(overrides["router"]["epp"]["flags"]["tracing"], "true")
                self.assertEqual(overrides["router"]["tracing"], {"enabled": True})

    def test_boolean_tracing_enabled(self):
        overrides = self.deploy({"router": {"epp": {"flags": {"tracing": True}}}})
        self.assertEqual(overrides["router"]["epp"]["flags"]["tracing"], "true")
        self.assertEqual(overrides["router"]["tracing"], {"enabled": True})

    def test_explicit_tracing_disabled(self):
        for value in ("false", False):
            with self.subTest(value=value):
                overrides = self.deploy(
                    {"router": {"epp": {"flags": {"tracing": value}}}}
                )
                actual = overrides["router"]["epp"]["flags"]["tracing"]
                self.assertEqual(actual, "false")
                self.assertEqual(overrides["router"]["tracing"], {"enabled": False})

    def test_chart_tracing_setting_takes_precedence(self):
        for enabled in (True, False):
            for flags in ({}, {"tracing": str(not enabled).lower()}):
                with self.subTest(enabled=enabled, flags=flags):
                    overrides = self.deploy(
                        {
                            "router": {
                                "tracing": {"enabled": enabled},
                                "epp": {"flags": flags},
                            }
                        }
                    )
                    self.assertEqual(
                        overrides["router"]["tracing"], {"enabled": enabled}
                    )
                    self.assertEqual(
                        overrides["router"]["epp"]["flags"]["tracing"],
                        str(enabled).lower(),
                    )

    def test_legacy_flag_boolean_spellings(self):
        for enabled, values in (
            (True, (True, 1, "1", "t", "T", "true", "TRUE", "True")),
            (False, (False, 0, "0", "f", "F", "false", "FALSE", "False")),
        ):
            for value in values:
                with self.subTest(value=value):
                    overrides = self.deploy(
                        {"router": {"epp": {"flags": {"tracing": value}}}}
                    )
                    self.assertEqual(
                        overrides["router"]["tracing"], {"enabled": enabled}
                    )
                    self.assertEqual(
                        overrides["router"]["epp"]["flags"]["tracing"],
                        str(enabled).lower(),
                    )

    def test_invalid_tracing_is_rejected_before_cluster_commands(self):
        for config in (
            {"router": {"tracing": {"enabled": "false"}}},
            {"router": {"tracing": {"enabled": 1}}},
            {"router": {"epp": {"flags": {"tracing": "invalid"}}}},
            {"router": {"epp": {"flags": {"tracing": []}}}},
        ):
            with self.subTest(config=config):
                config_path = self.directory / "invalid.yaml"
                config_path.write_text(yaml.safe_dump(config))
                with (
                    mock.patch.object(perf, "run_cmd") as run_cmd,
                    mock.patch.object(
                        perf,
                        "open",
                        mock.mock_open(read_data=yaml.safe_dump(config)),
                        create=True,
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "tracing"):
                        perf.deploy_epp(
                            "offline-invalid", "unused", "unused", str(config_path)
                        )
                    run_cmd.assert_not_called()

    def test_invalid_yaml_is_rejected_before_helm(self):
        config_path = self.directory / "invalid.yaml"
        config_path.write_text("router: [")
        with (
            mock.patch.object(perf, "run_cmd") as run_cmd,
            mock.patch.object(
                perf, "open", mock.mock_open(read_data="router: ["), create=True
            ),
        ):
            with self.assertRaises(yaml.YAMLError):
                perf.deploy_epp("offline-invalid", "unused", "unused", str(config_path))
            run_cmd.assert_not_called()

    def test_missing_or_null_tracing_remains_disabled(self):
        for config in (
            {},
            {"router": {}},
            {"router": {"epp": {}}},
            {"router": {"epp": {"flags": {}}}},
            {"router": {"epp": {"flags": {"tracing": None}}}},
            {"router": {"tracing": {"enabled": None}}},
        ):
            with self.subTest(config=config):
                overrides = self.deploy(config)
                self.assertEqual(
                    overrides["router"]["epp"]["flags"]["tracing"], "false"
                )
                self.assertEqual(overrides["router"]["tracing"], {"enabled": False})

    def test_existing_router_recipes_remain_disabled(self):
        config_dir = Path(perf.__file__).parent / "config" / "router-configs"
        recipes = sorted(config_dir.glob("*.yaml"))
        self.assertTrue(recipes)
        for recipe in recipes:
            with self.subTest(recipe=recipe.name):
                config = yaml.safe_load(recipe.read_text())
                overrides = self.deploy(config)
                self.assertEqual(
                    overrides["router"]["epp"]["flags"]["tracing"], "false"
                )

    def test_other_benchmark_overrides_are_preserved(self):
        config = {
            "router": {
                "epp": {"flags": {"tracing": "true", "v": 9}},
                "modelServers": {"matchLabels": {"app": "model", "role": "decode"}},
            }
        }
        with mock.patch.dict(
            os.environ,
            {
                "EPP_REGISTRY": "registry.example",
                "EPP_REPOSITORY": "epp",
                "EPP_TAG": "pinned",
            },
        ):
            overrides = self.deploy(
                config,
                epp_cpu="750m",
                epp_memory="3Gi",
                epp_replicas=2,
                machine_family="c3",
            )
        router = overrides["router"]
        epp = router["epp"]
        self.assertEqual(
            epp["flags"], {"v": 4, "enable-pprof": "true", "tracing": "true"}
        )
        self.assertEqual(
            epp["image"],
            {
                "registry": "registry.example",
                "repository": "epp",
                "tag": "pinned",
            },
        )
        self.assertEqual(epp["replicas"], 2)
        self.assertEqual(
            epp["resources"],
            {
                "requests": {"cpu": "750m", "memory": "3Gi"},
                "limits": {"cpu": "1500m", "memory": "6Gi"},
            },
        )
        affinity = epp["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]
        self.assertEqual(
            affinity["nodeSelectorTerms"][0]["matchExpressions"][0]["values"], ["c3"]
        )
        self.assertEqual(
            router["modelServers"]["matchLabels"], {"app": "llm-d-sim", "role": None}
        )
        self.assertFalse(router["monitoring"]["prometheus"]["auth"]["enabled"])
        self.assertTrue(router["proxy"]["enabled"])


class PercentileTests(unittest.TestCase):
    def test_scheduler_delta_percentiles_in_milliseconds(self):
        before, after = scheduler_metrics()
        original = copy.deepcopy((before, after))
        result = perf.calculate_percentiles(before, after)
        self.assertEqual(len(result), 3)
        for actual, expected in zip(result, (1.0, 55.0, 91.0)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual((before, after), original)

    def test_missing_metrics_return_three_zeros(self):
        before, after = scheduler_metrics()
        for first, second in ((None, after), (before, None), ({}, after), (before, {})):
            with self.subTest(before=first, after=second):
                self.assertEqual(
                    perf.calculate_percentiles(first, second), (0.0, 0.0, 0.0)
                )

    def test_no_new_events_return_three_zeros(self):
        before, _ = scheduler_metrics()
        for count in (100, 0):
            with self.subTest(count=count):
                after = {"count": count, "buckets": before["buckets"]}
                self.assertEqual(
                    perf.calculate_percentiles(before, after), (0.0, 0.0, 0.0)
                )

    def test_newly_observed_buckets(self):
        _, after = scheduler_metrics()
        before = {"count": 0, "buckets": {}}
        result = perf.calculate_percentiles(before, after)
        self.assertEqual(len(result), 3)
        for actual, expected in zip(result, (1.0, 55.0, 91.0)):
            self.assertAlmostEqual(actual, expected)

    def test_unbounded_bucket_uses_existing_upper_bound_convention(self):
        before = {"count": 0, "buckets": {}}
        after = {"count": 100, "buckets": {"0.001": 80, "0.01": 90, "+Inf": 100}}
        self.assertEqual(perf.calculate_percentiles(before, after), (0.625, 10.0, 10.0))


class MarkdownReportTests(OfflineTestCase):
    def write_report(self, **kwargs):
        perf.write_results_to_markdown_folder(
            str(self.directory),
            "tracing",
            "2026-09-07 00:00:00",
            "offline",
            "config/router.yaml",
            "config/perf.yaml",
            "c3",
            10,
            ["epp:pinned"],
            {"epp": {"cpu": 10, "mem": 20}},
            {"epp": {"cpu": 30, "mem": 40}},
            p50=1.0,
            p95=55.0,
            p99=91.0,
            status="SUCCESS",
            **kwargs,
        )
        return (self.directory / "tracing.md").read_text()

    def test_report_contains_p99_in_aligned_column(self):
        report = self.write_report()
        lines = [line for line in report.splitlines() if line.startswith("|")]
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(len(line.strip("|").split("|")) == 18 for line in lines))
        self.assertIn("P99 Latency (ms)", lines[0])
        self.assertEqual(lines[2].split("|")[15].strip(), "91.00")
        self.assertEqual(lines[2].split("|")[-2].strip(), "SUCCESS")

    def test_repeated_append_uses_one_header(self):
        first = self.write_report()
        second = self.write_report()
        self.assertTrue(second.startswith(first))
        self.assertEqual(second.count("| Timestamp |"), 1)
        self.assertEqual(second.count("| 91.00 |"), 2)

    def test_legacy_table_is_preserved_and_new_table_is_appended(self):
        legacy = "# Existing results\n\n" + LEGACY_HEADER
        legacy += "|" + "---|" * 17 + "\n"
        legacy += "| " + " | ".join(["historical"] * 17) + " |\n"
        (self.directory / "tracing.md").write_text(legacy)
        report = self.write_report()
        self.assertTrue(report.startswith(legacy))
        self.assertEqual(report.count("| Timestamp |"), 2)
        self.assertIn("\n\n| Timestamp |", report[len(legacy) - 1 :])
        self.assertIn("P99 Latency (ms)", report[len(legacy) :])
        self.assertEqual(self.write_report().count("| Timestamp |"), 2)

    def test_profile_links_and_status_follow_p99(self):
        profile = {
            name: str(self.directory / f"{name}.profile")
            for name in (
                "cpu_svg",
                "cpu_png",
                "cpu_pprof",
                "mem_svg",
                "mem_png",
                "mem_pprof",
            )
        }
        report = self.write_report(profile_results=profile)
        row = report.splitlines()[-1].split("|")
        self.assertEqual(row[15].strip(), "91.00")
        self.assertIn("cpu_pprof.profile", row[16])
        self.assertIn("mem_pprof.profile", row[17])
        self.assertEqual(row[18].strip(), "SUCCESS")


class MainWiringTests(OfflineTestCase):
    def test_main_passes_scheduler_p99_to_report(self):
        for name in (
            "create_namespace",
            "setup_hf_secret",
            "setup_perf_sa",
            "deploy_simulators",
            "deploy_epp",
            "run_benchmark",
            "cleanup_namespace",
        ):
            self.contexts.enter_context(mock.patch.object(perf, name))
        self.contexts.enter_context(mock.patch.object(perf, "Thread"))
        self.contexts.enter_context(mock.patch.object(perf.time, "sleep"))
        self.contexts.enter_context(
            mock.patch.object(perf, "get_epp_pod_name", return_value="epp-pod")
        )
        self.contexts.enter_context(
            mock.patch.object(perf, "get_container_images", return_value=["epp:pinned"])
        )
        self.contexts.enter_context(
            mock.patch.object(
                perf, "sample_resources", return_value={"epp": {"cpu": 1, "mem": 2}}
            )
        )
        self.contexts.enter_context(
            mock.patch.object(
                perf, "scrape_scheduler_metrics", side_effect=scheduler_metrics()
            )
        )
        writer = self.contexts.enter_context(
            mock.patch.object(perf, "write_results_to_markdown_folder", autospec=True)
        )
        self.contexts.enter_context(
            mock.patch.object(
                perf.sys,
                "argv",
                [
                    "run_nightly_perf.py",
                    "--gcp-project",
                    "offline-project",
                    "--results-dir",
                    str(self.directory),
                ],
            )
        )
        self.contexts.enter_context(mock.patch.object(perf, "stop_monitoring", False))
        perf.main()
        writer.assert_called_once()
        self.assertAlmostEqual(writer.call_args.args[13], 91.0)
        self.assertEqual(writer.call_args.args[14], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
