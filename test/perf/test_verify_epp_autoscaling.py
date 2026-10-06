import unittest

import verify_epp_autoscaling as hpa_test


SAMPLE_TOP_OUTPUT = """\
load-aware-hpa-epp-aaa111   envoy-proxy   410m   58Mi
load-aware-hpa-epp-aaa111   epp           820m   112Mi
load-aware-hpa-epp-bbb222   envoy-proxy   390m   60Mi
load-aware-hpa-epp-bbb222   epp           780m   108Mi
"""

SAMPLE_HPA_JSON = {
    "status": {
        "currentReplicas": 3,
        "desiredReplicas": 3,
        "currentMetrics": [
            {
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "current": {
                        "averageUtilization": 142,
                        "averageValue": "710m",
                    },
                },
            }
        ],
        "conditions": [
            {"type": "AbleToScale", "status": "True", "reason": "ReadyForNewScale"},
            {"type": "ScalingActive", "status": "True", "reason": "ValidMetricFound"},
        ],
    },
    "spec": {
        "minReplicas": 1,
        "maxReplicas": 4,
        "metrics": [
            {
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": 70,
                    },
                },
            }
        ],
    },
}


class ParseMetricsTests(unittest.TestCase):
    def test_parse_pod_top_output_computes_per_pod_and_epp_totals(self):
        parsed = hpa_test.parse_top_containers_output(SAMPLE_TOP_OUTPUT)
        self.assertEqual(parsed["pod_count"], 2)
        self.assertEqual(parsed["containers"]["epp"]["cpu_m"], 1600)
        self.assertEqual(parsed["containers"]["epp"]["mem_mib"], 220)
        self.assertEqual(parsed["containers"]["envoy-proxy"]["cpu_m"], 800)
        self.assertEqual(parsed["containers"]["envoy-proxy"]["mem_mib"], 118)
        self.assertEqual(parsed["total_cpu_m"], 2400)
        self.assertEqual(parsed["total_mem_mib"], 338)
        self.assertEqual(
            parsed["pods"]["load-aware-hpa-epp-aaa111"]["epp"]["cpu_m"], 820
        )
        self.assertEqual(
            parsed["pods"]["load-aware-hpa-epp-bbb222"]["epp"]["cpu_m"], 780
        )

    def test_parse_hpa_status_extracts_replicas_and_cpu_utilization(self):
        status = hpa_test.parse_hpa_object(SAMPLE_HPA_JSON)
        self.assertEqual(status["current_replicas"], 3)
        self.assertEqual(status["desired_replicas"], 3)
        self.assertEqual(status["min_replicas"], 1)
        self.assertEqual(status["max_replicas"], 4)
        self.assertEqual(status["current_cpu_utilization_pct"], 142)
        self.assertEqual(status["current_cpu_average_value"], "710m")
        self.assertEqual(status["target_cpu_utilization_pct"], 70)
        self.assertTrue(status["scaling_active"])


class LogAuditTests(unittest.TestCase):
    def test_audit_epp_logs_passes_clean_json_logs(self):
        logs = "\n".join(
            [
                '{"level":"info","ts":1700000,"msg":"Starting server"}',
                '{"level":"debug","ts":1700001,"msg":"Scheduled request","pod":"10.0.0.1"}',
            ]
        )
        errors = hpa_test.find_epp_log_errors(logs)
        self.assertEqual(errors, [])

    def test_audit_epp_logs_detects_error_and_panic_lines(self):
        logs = "\n".join(
            [
                '{"level":"info","ts":1700000,"msg":"Starting server"}',
                '{"level":"error","ts":1700002,"msg":"Failed to route request"}',
                "panic: runtime error: invalid memory address",
            ]
        )
        errors = hpa_test.find_epp_log_errors(logs)
        self.assertEqual(len(errors), 2)

    def test_parse_inference_perf_logs_verifies_zero_failures(self):
        perf_log = """\
2026-10-06 21:00:00 INFO Stage 1 - run started
2026-10-06 21:00:30 INFO Stage 1 - run completed
2026-10-06 21:00:31 INFO Stage 2 - run started
2026-10-06 21:03:31 INFO Stage 2 - run completed
2026-10-06 21:03:32 INFO Stage 3 - run started
2026-10-06 21:06:32 INFO Stage 3 - run completed
"failed_requests": 0,
"successful_requests": 2400,
"""
        summary = hpa_test.audit_inference_perf_logs(perf_log, expected_stages=3)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["completed_stages"], 3)
        self.assertEqual(summary["failed_requests"], 0)
        self.assertEqual(summary["error_lines"], [])

    def test_parse_inference_perf_logs_flags_failed_requests(self):
        perf_log = """\
2026-10-06 21:00:00 INFO Stage 1 - run started
2026-10-06 21:00:30 INFO Stage 1 - run completed
"failed_requests": 4,
"""
        summary = hpa_test.audit_inference_perf_logs(perf_log, expected_stages=1)
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["failed_requests"], 4)


class ScalingVerificationTests(unittest.TestCase):
    def make_sample(
        self,
        ts,
        current_replicas,
        desired_replicas,
        ready_replicas,
        hpa_cpu_pct,
        epp_cpu_m,
        envoy_cpu_m=200,
        restarts=0,
    ):
        return {
            "timestamp": ts,
            "hpa": {
                "current_replicas": current_replicas,
                "desired_replicas": desired_replicas,
                "current_cpu_utilization_pct": hpa_cpu_pct,
                "target_cpu_utilization_pct": 70,
                "scaling_active": True,
            },
            "deployment": {
                "ready_replicas": ready_replicas,
                "restarts": restarts,
            },
            "resources": {
                "pod_count": ready_replicas,
                "total_cpu_m": epp_cpu_m + envoy_cpu_m,
                "total_mem_mib": 150 * ready_replicas,
                "containers": {
                    "epp": {"cpu_m": epp_cpu_m, "mem_mib": 100 * ready_replicas},
                    "envoy-proxy": {
                        "cpu_m": envoy_cpu_m,
                        "mem_mib": 50 * ready_replicas,
                    },
                },
                "pods": {},
            },
        }

    def test_evaluate_autoscaling_passes_when_scale_up_and_down_and_zero_errors(self):
        timeline = [
            self.make_sample("00:00", 1, 1, 1, 26, 110, 20),
            self.make_sample("01:00", 1, 3, 1, 320, 1250, 650),
            self.make_sample("02:00", 3, 3, 3, 115, 1320, 700),
            self.make_sample("05:00", 3, 1, 3, 22, 300, 60),
            self.make_sample("06:00", 1, 1, 1, 28, 120, 20),
        ]
        verdict = hpa_test.evaluate_autoscaling_run(
            timeline,
            min_replicas=1,
            target_cpu_pct=70,
            epp_log_errors={},
            perf_audit={"passed": True, "failed_requests": 0, "error_lines": []},
        )
        self.assertTrue(verdict["passed"], verdict["failures"])
        self.assertTrue(verdict["scale_up_verified"])
        self.assertTrue(verdict["scale_down_verified"])
        self.assertTrue(verdict["epp_cpu_metrics_verified"])
        self.assertEqual(verdict["initial_replicas"], 1)
        self.assertEqual(verdict["peak_replicas"], 3)
        self.assertEqual(verdict["final_replicas"], 1)
        self.assertEqual(verdict["peak_epp_cpu_m"], 1320)
        self.assertEqual(verdict["peak_hpa_cpu_pct"], 320)

    def test_evaluate_autoscaling_fails_if_no_scale_up(self):
        timeline = [
            self.make_sample("00:00", 1, 1, 1, 26, 110, 20),
            self.make_sample("02:00", 1, 1, 1, 55, 200, 50),
        ]
        verdict = hpa_test.evaluate_autoscaling_run(
            timeline,
            min_replicas=1,
            target_cpu_pct=70,
            epp_log_errors={},
            perf_audit={"passed": True, "failed_requests": 0, "error_lines": []},
        )
        self.assertFalse(verdict["passed"])
        self.assertFalse(verdict["scale_up_verified"])

    def test_evaluate_autoscaling_fails_if_no_scale_down(self):
        timeline = [
            self.make_sample("00:00", 1, 1, 1, 26, 110, 20),
            self.make_sample("02:00", 3, 3, 3, 180, 1300, 650),
            self.make_sample("05:00", 3, 3, 3, 90, 900, 450),
        ]
        verdict = hpa_test.evaluate_autoscaling_run(
            timeline,
            min_replicas=1,
            target_cpu_pct=70,
            epp_log_errors={},
            perf_audit={"passed": True, "failed_requests": 0, "error_lines": []},
        )
        self.assertFalse(verdict["passed"])
        self.assertTrue(verdict["scale_up_verified"])
        self.assertFalse(verdict["scale_down_verified"])


if __name__ == "__main__":
    unittest.main()
