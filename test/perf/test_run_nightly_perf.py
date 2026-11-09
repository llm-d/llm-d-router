import argparse
from contextlib import ExitStack
import copy
import json
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
        proxy_mode = ""
        if isinstance(config.get("router"), dict) and isinstance(
            config["router"].get("proxy"), dict
        ):
            proxy_mode = str(config["router"]["proxy"].get("mode") or "").lower()
        expected_calls = 3 if proxy_mode == "service" else 2
        self.assertEqual(run_cmd.call_count, expected_calls)
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
        configs = [
            {"router": {"tracing": {"enabled": "false"}}},
            {"router": {"tracing": {"enabled": 1}}},
            {"router": {"epp": {"flags": {"tracing": "invalid"}}}},
            {"router": {"epp": {"flags": {"tracing": []}}}},
        ]
        for value in (True, False, "invalid", "", 1, 0, ["invalid"], []):
            configs.extend(
                [
                    {"router": {"tracing": value}},
                    {"router": {"epp": value}},
                    {"router": {"epp": {"flags": value}}},
                ]
            )
        for config in configs:
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
            {"router": {"tracing": None}},
            {"router": {"epp": None}},
            {"router": {"epp": {}}},
            {"router": {"epp": {"flags": None}}},
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

    def test_malformed_model_server_labels_warn_and_keep_simulator_overrides(self):
        for model_servers in (
            True,
            1,
            "invalid",
            ["matchLabels"],
            {"matchLabels": True},
            {"matchLabels": 1},
            {"matchLabels": ["role"]},
        ):
            with self.subTest(model_servers=model_servers):
                with mock.patch.object(perf, "print") as output:
                    overrides = self.deploy(
                        {
                            "router": {
                                "tracing": {"enabled": True},
                                "modelServers": model_servers,
                            }
                        }
                    )
                self.assertEqual(
                    overrides["router"]["modelServers"]["matchLabels"],
                    {"app": "llm-d-sim"},
                )
                self.assertEqual(overrides["router"]["tracing"], {"enabled": True})
                warnings = [
                    call.args[0]
                    for call in output.call_args_list
                    if call.args[0].startswith("Warning:")
                ]
                self.assertEqual(len(warnings), 1)
                self.assertIn("modelServers labels for nullification", warnings[0])

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

    def test_autoscaling_enabled_omits_static_replicas_override(self):
        config = {
            "router": {
                "epp": {
                    "autoscaling": {
                        "enabled": True,
                        "minReplicas": 1,
                        "maxReplicas": 4,
                    }
                }
            }
        }
        overrides = self.deploy(config, epp_replicas=2)
        self.assertNotIn("replicas", overrides["router"]["epp"])

    def test_proxy_service_mode_waits_for_proxy_deployment_rollout(self):
        config = {
            "router": {
                "proxy": {
                    "mode": "service",
                    "autoscaling": {
                        "enabled": True,
                        "minReplicas": 1,
                        "maxReplicas": 4,
                    },
                }
            }
        }
        overrides = self.deploy(config, epp_replicas=1)
        self.assertEqual(overrides["router"]["epp"]["replicas"], 1)
        self.assertTrue(overrides["router"]["proxy"]["enabled"])

    def test_run_benchmark_uses_custom_service_name(self):
        job_path = self.directory / "job.yaml"
        job_path.write_text(
            yaml.safe_dump(
                {
                    "config": {"server": {"base_url": "http://old:80"}},
                    "token": {"hfSecret": {"name": "old", "key": "old"}},
                    "job": {"serviceAccountName": "old"},
                }
            )
        )
        out_job_path = self.directory / "temp-job-values.yaml"
        real_open = open

        def local_open(path, *args, **open_kwargs):
            if os.fspath(path) == "/tmp/temp-job-values.yaml":
                path = out_job_path
            return real_open(path, *args, **open_kwargs)

        def fake_run_cmd(cmd, check=True):
            res = mock.MagicMock()
            res.returncode = 0
            res.stdout = "inference-perf-job-abc" if "get pods" in str(cmd) else ""
            return res

        with (
            mock.patch.object(perf, "open", local_open, create=True),
            mock.patch.object(perf, "run_cmd", side_effect=fake_run_cmd),
        ):
            perf.run_benchmark(
                "test-ns",
                str(job_path),
                "/chart",
                "my-release",
                service_name="my-release-proxy",
            )
        rendered = yaml.safe_load(out_job_path.read_text())
        self.assertEqual(
            rendered["config"]["server"]["base_url"], "http://my-release-proxy:80"
        )


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


class MeanLatencyTests(unittest.TestCase):
    def test_mean_from_sum_and_count_deltas_in_milliseconds(self):
        before = {"count": 100, "sum": 0.5, "buckets": {}}
        after = {"count": 300, "sum": 1.5, "buckets": {}}
        self.assertAlmostEqual(perf.calculate_mean_latency(before, after), 5.0)

    def test_missing_metrics_or_no_events_return_zero(self):
        before, after = scheduler_metrics()
        for first, second in ((None, after), (before, None), (before, before)):
            with self.subTest(before=first, after=second):
                self.assertEqual(perf.calculate_mean_latency(first, second), 0.0)


class ProvenanceParsingTests(unittest.TestCase):
    def test_epp_info_commit_with_any_label_order(self):
        for line in (
            'llm_d_epp_info{build_ref="main",commit="abc123"} 1',
            'llm_d_epp_info{commit="abc123",build_ref="main"} 1',
            'llm_d_router_epp_info{build_ref="main",commit="abc123"} 1',
        ):
            with self.subTest(line=line):
                self.assertEqual(perf.parse_epp_info_commit(line), "abc123")

    def test_epp_info_without_commit_label_is_empty(self):
        self.assertEqual(perf.parse_epp_info_commit('llm_d_epp_info{build_ref="main"} 1'), "")

    def test_other_metrics_are_not_epp_info(self):
        self.assertIsNone(perf.parse_epp_info_commit("llm_d_epp_scheduler_e2e_duration_seconds_count 5"))

    def test_container_image_ids(self):
        output = "epp=ghcr.io/llm-d/epp@sha256:aaa\nenvoy-proxy=docker.io/envoyproxy/envoy@sha256:bbb\n"
        self.assertEqual(
            perf.parse_container_image_ids(output),
            {
                "epp": "ghcr.io/llm-d/epp@sha256:aaa",
                "envoy-proxy": "docker.io/envoyproxy/envoy@sha256:bbb",
            },
        )

    def test_empty_image_id_output(self):
        self.assertEqual(perf.parse_container_image_ids(""), {})


class IdleSampleTests(unittest.TestCase):
    def test_container_missing_from_the_first_sample_is_kept(self):
        samples = [
            {"envoy-proxy": {"cpu": 10, "mem": 20}},
            {"envoy-proxy": {"cpu": 30, "mem": 20}, "epp": {"cpu": 100, "mem": 40}},
            {"envoy-proxy": {"cpu": 20, "mem": 20}, "epp": {"cpu": 200, "mem": 60}},
        ]
        self.assertEqual(
            perf.average_idle_samples(samples),
            {"envoy-proxy": {"cpu": 20, "mem": 20}, "epp": {"cpu": 150, "mem": 50}},
        )


class RunRecordTests(OfflineTestCase):
    def record(self, **overrides):
        values = dict(
            test_name="optimized-baseline-job1",
            started_at="2026-10-19T09:00:00Z",
            ns="llm-d-perf-1",
            status="SUCCESS",
            target={"id": "optimized-baseline-job1", "digest": "d1", "setup": {"sim_replicas": 10}},
            epp_commit="abc123",
            images=["epp:main"],
            image_ids={"epp": "epp@sha256:aaa"},
            idle_metrics={"epp": {"cpu": 10, "mem": 20}},
            peak_metrics={"epp": {"cpu": 30, "mem": 40}, "TOTAL": {"cpu": 50, "mem": 60}},
            latency={"p50": 1.0, "p95": 2.0, "p99": 3.0, "mean": 1.5},
        )
        values.update(overrides)
        return perf.build_run_record(**values)

    def test_record_contents(self):
        record = self.record()
        self.assertEqual(record["schema_version"], perf.RUN_RECORD_SCHEMA_VERSION)
        self.assertEqual(record["target"]["digest"], "d1")
        self.assertEqual(record["provenance"]["epp_commit"], "abc123")
        self.assertEqual(record["provenance"]["image_ids"], {"epp": "epp@sha256:aaa"})
        self.assertEqual(
            record["containers"]["epp"],
            {"idle_cpu_m": 10, "idle_mem_mib": 20, "peak_cpu_m": 30, "peak_mem_mib": 40},
        )
        self.assertEqual(record["containers"]["TOTAL"]["idle_cpu_m"], None)
        self.assertEqual(record["scheduler_latency_ms"]["mean"], 1.5)

    def test_record_is_written_under_runs_by_test_name(self):
        path = perf.write_run_record(str(self.directory), self.record())
        self.assertEqual(
            Path(path),
            self.directory / "runs" / "optimized-baseline-job1" / "20261019T090000Z.json",
        )
        self.assertEqual(json.loads(Path(path).read_text()), self.record())

    def test_paths_inside_the_repo_are_made_relative(self):
        repo = str(self.directory)
        inside = os.path.join(repo, "test", "perf", "job.yaml")
        self.assertEqual(perf.repo_relative_path(inside, repo), os.path.join("test", "perf", "job.yaml"))
        self.assertEqual(perf.repo_relative_path("/elsewhere/job.yaml", repo), "/elsewhere/job.yaml")


class TargetTests(OfflineTestCase):
    def write(self, relative, content):
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return str(path)

    def repo_with_target(self, trend="metrics: []", epp_cpu="'2'", epp_memory="4Gi"):
        for name in ("router.yaml", "job.yaml", "sim.yaml", "svc.yaml"):
            self.write(f"config/{name}", f"{name} contents\n")
        return self.write(
            "targets/bench.yaml",
            "setup:\n"
            "  router_config: config/router.yaml\n"
            "  perf_job: config/job.yaml\n"
            "  sim_deploy: config/sim.yaml\n"
            "  sim_svc: config/svc.yaml\n"
            "  sim_replicas: 3\n"
            "  epp_replicas: 1\n"
            f"  epp_cpu: {epp_cpu}\n"
            f"  epp_memory: {epp_memory}\n"
            "  router_machine_family: c3\n"
            "  epp_log_verbosity: 2\n"
            f"trend: {{{trend}}}\n",
        )

    def target_for(self, target_path):
        args = argparse.Namespace()
        perf.apply_target(args, target_path, str(self.directory))
        return perf.build_target(args, str(self.directory))

    def test_nightly_target_matches_the_workflow_flags(self):
        repo = Path(perf.__file__).resolve().parents[2]
        args = argparse.Namespace()
        perf.apply_target(args, str(repo / "test/perf/config/targets/optimized-baseline-job1.yaml"), str(repo))
        self.assertEqual(args.test_name, "optimized-baseline-job1")
        self.assertEqual(args.router_config, str(repo / "test/perf/config/router-configs/optimized-baseline.yaml"))
        self.assertEqual(args.perf_job, str(repo / "test/perf/config/shared_prefix_job1.yaml"))
        self.assertEqual((args.sim_replicas, args.epp_replicas, args.router_machine_family), (10, 1, "e2"))
        self.assertEqual(args.epp_log_verbosity, perf.EPP_LOG_VERBOSITY)

    def test_target_setup_is_repository_relative(self):
        target = self.target_for(self.repo_with_target())
        self.assertEqual(target["id"], "bench")
        self.assertEqual(target["setup"]["perf_job"], os.path.join("config", "job.yaml"))
        self.assertEqual(target["setup"]["router_machine_family"], "c3")

    def test_unknown_setup_key_or_section_is_rejected(self):
        for content in ("setup:\n  sim_replica: 3\n", "setups:\n  sim_replicas: 3\n"):
            with self.subTest(content=content):
                target = self.write("targets/bad.yaml", content)
                with self.assertRaises(ValueError):
                    perf.load_target(target)

    def test_unquoted_resources_are_read_as_flag_strings(self):
        quoted = self.target_for(self.repo_with_target(epp_cpu="'2'", epp_memory="'4096'"))
        args = argparse.Namespace()
        target = self.repo_with_target(epp_cpu="2", epp_memory="4096")
        perf.apply_target(args, target, str(self.directory))
        self.assertEqual((args.epp_cpu, args.epp_memory), ("2", "4096"))
        self.assertEqual((perf.double_cpu(args.epp_cpu), perf.double_memory(args.epp_memory)), ("4", "8192"))
        self.assertEqual(self.target_for(target)["digest"], quoted["digest"])

    def test_setup_values_of_the_wrong_type_are_rejected(self):
        for key, value in (("sim_replicas", "ten"), ("sim_replicas", 2.5), ("epp_cpu", "true"), ("router_config", "[a, b]")):
            with self.subTest(key=key, value=value):
                target = self.write("targets/bad.yaml", f"setup:\n  {key}: {value}\n")
                with self.assertRaises(ValueError):
                    perf.load_target(target)

    def test_digest_changes_when_a_referenced_file_is_edited_in_place(self):
        target = self.repo_with_target()
        before = self.target_for(target)["digest"]
        self.write("config/job.yaml", "edited stages\n")
        self.assertNotEqual(self.target_for(target)["digest"], before)

    def test_router_config_edit_keeps_the_digest_and_is_recorded(self):
        target = self.repo_with_target()
        before = self.target_for(target)
        self.write("config/router.yaml", "edited scorer weights\n")
        after = self.target_for(target)
        self.assertEqual(after["digest"], before["digest"])
        self.assertNotEqual(after["router_config_sha256"], before["router_config_sha256"])

    def test_digest_ignores_the_trend_section(self):
        first = self.target_for(self.repo_with_target("metrics: []"))["digest"]
        second = self.target_for(self.repo_with_target("window_days: 14"))["digest"]
        self.assertEqual(first, second)


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
        mocks = {}
        for name in (
            "create_namespace",
            "setup_hf_secret",
            "setup_perf_sa",
            "deploy_simulators",
            "deploy_epp",
            "run_benchmark",
            "cleanup_namespace",
        ):
            mocks[name] = self.contexts.enter_context(mock.patch.object(perf, name))
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
                perf, "get_container_image_ids", return_value={"epp": "epp@sha256:aaa"}
            )
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
                    "--target",
                    str(Path(perf.__file__).resolve().parent / "config" / "targets" / "optimized-baseline-job1.yaml"),
                ],
            )
        )
        self.contexts.enter_context(mock.patch.object(perf, "stop_monitoring", False))
        perf.main()
        deploy = mocks["deploy_epp"].call_args.kwargs
        self.assertEqual((deploy["log_verbosity"], deploy["machine_family"]), (perf.EPP_LOG_VERBOSITY, "e2"))
        writer.assert_called_once()
        self.assertAlmostEqual(writer.call_args.args[13], 91.0)
        self.assertEqual(writer.call_args.args[14], "SUCCESS")
        records = list((self.directory / "runs" / "optimized-baseline-job1").glob("*.json"))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text())
        self.assertEqual(record["status"], "SUCCESS")
        self.assertEqual(record["provenance"]["image_ids"], {"epp": "epp@sha256:aaa"})
        self.assertAlmostEqual(record["scheduler_latency_ms"]["p99"], 91.0)
        self.assertEqual(record["target"]["id"], "optimized-baseline-job1")
        self.assertEqual(len(record["target"]["digest"]), 64)
        self.assertEqual(record["target"]["setup"]["epp_log_verbosity"], perf.EPP_LOG_VERBOSITY)
        self.assertEqual(
            record["target"]["setup"]["router_config"],
            os.path.join("test", "perf", "config", "router-configs", "optimized-baseline.yaml"),
        )


if __name__ == "__main__":
    unittest.main()
