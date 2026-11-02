#!/usr/bin/env python3
"""End-to-end benchmark and verification script for EPP Horizontal Pod Autoscaling (HPA).

Deploys vLLM simulators and EPP in active-active mode with HPA enabled, drives
multi-stage inference-perf traffic, continuously samples HPA status, replica counts,
and per-pod/per-container EPP CPU/memory metrics, audits all EPP/Envoy/inference-perf
logs for zero errors, and verifies both scale-up and scale-down occur.
"""

import argparse
import json
import os
import re
import sys
import time
from threading import Thread
import yaml

import run_nightly_perf as perf


def parse_cpu_millicores(cpu_str):
    cpu_str = str(cpu_str).strip()
    if not cpu_str:
        return 0
    if cpu_str.endswith("n"):
        return int(round(float(cpu_str[:-1]) / 1_000_000.0))
    if cpu_str.endswith("u"):
        return int(round(float(cpu_str[:-1]) / 1_000.0))
    if cpu_str.endswith("m"):
        return int(float(cpu_str[:-1]))
    return int(round(float(cpu_str) * 1000))


def parse_mem_mib(mem_str):
    mem_str = str(mem_str).strip()
    if not mem_str:
        return 0
    if mem_str.endswith("Ki"):
        return int(round(float(mem_str[:-2]) / 1024.0))
    if mem_str.endswith("Mi"):
        return int(float(mem_str[:-2]))
    if mem_str.endswith("Gi"):
        return int(round(float(mem_str[:-2]) * 1024.0))
    return int(mem_str)


def parse_top_containers_output(stdout_text):
    """Parses `kubectl top pod -l ... --containers --no-headers` output."""
    pods = {}
    containers = {}
    total_cpu_m = 0
    total_mem_mib = 0

    for raw_line in (stdout_text or "").strip().splitlines():
        parts = raw_line.split()
        if len(parts) < 4:
            continue
        pod_name, container_name, cpu_str, mem_str = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
        )
        cpu_m = parse_cpu_millicores(cpu_str)
        mem_mib = parse_mem_mib(mem_str)

        pods.setdefault(pod_name, {})[container_name] = {
            "cpu_m": cpu_m,
            "mem_mib": mem_mib,
        }
        agg = containers.setdefault(container_name, {"cpu_m": 0, "mem_mib": 0})
        agg["cpu_m"] += cpu_m
        agg["mem_mib"] += mem_mib
        total_cpu_m += cpu_m
        total_mem_mib += mem_mib

    return {
        "pod_count": len(pods),
        "pods": pods,
        "containers": containers,
        "total_cpu_m": total_cpu_m,
        "total_mem_mib": total_mem_mib,
    }


def parse_hpa_object(hpa_obj):
    """Extracts replica counts, CPU utilization, and conditions from an HPA v2 dict."""
    spec = (hpa_obj or {}).get("spec") or {}
    status = (hpa_obj or {}).get("status") or {}

    target_cpu_pct = None
    for metric in spec.get("metrics") or []:
        if metric.get("type") in ("Resource", "ContainerResource"):
            res = metric.get("resource") or metric.get("containerResource") or {}
            if res.get("name") == "cpu":
                target = res.get("target") or {}
                target_cpu_pct = target.get("averageUtilization")

    current_cpu_pct = None
    current_cpu_avg_val = None
    for metric in status.get("currentMetrics") or []:
        if metric.get("type") in ("Resource", "ContainerResource"):
            res = metric.get("resource") or metric.get("containerResource") or {}
            if res.get("name") == "cpu":
                current = res.get("current") or {}
                current_cpu_pct = current.get("averageUtilization")
                current_cpu_avg_val = current.get("averageValue")

    conditions = {}
    for cond in status.get("conditions") or []:
        ctype = cond.get("type")
        if ctype:
            conditions[ctype] = {
                "status": cond.get("status"),
                "reason": cond.get("reason", ""),
                "message": cond.get("message", ""),
            }

    scaling_active = (
        conditions.get("ScalingActive", {}).get("status") == "True"
    )

    return {
        "min_replicas": int(spec.get("minReplicas", 1)),
        "max_replicas": int(spec.get("maxReplicas", 1)),
        "current_replicas": int(status.get("currentReplicas", 0) or 0),
        "desired_replicas": int(status.get("desiredReplicas", 0) or 0),
        "current_cpu_utilization_pct": current_cpu_pct,
        "current_cpu_average_value": current_cpu_avg_val,
        "target_cpu_utilization_pct": target_cpu_pct,
        "scaling_active": scaling_active,
        "conditions": conditions,
    }


def find_epp_log_errors(log_text):
    """Scans EPP or Envoy container logs for error-level messages or panics."""
    error_lines = []
    benign_substrings = (
        "Failed to watch",
        "context canceled",
        "Server closed",
        "graceful stop",
        "terminating",
    )
    for raw_line in (log_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(b in line for b in benign_substrings):
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                level = str(parsed.get("level", "")).lower()
                if level in ("error", "dpanic", "panic", "fatal"):
                    error_lines.append(line)
                continue
            except json.JSONDecodeError:
                pass
        if (
            line.startswith("panic:")
            or line.startswith("fatal error:")
            or re.match(r"^E\d{4}\s", line)
            or '"level":"error"' in line.lower()
        ):
            error_lines.append(line)
    return error_lines


def audit_inference_perf_logs(log_text, expected_stages=1):
    """Parses inference-perf logs to verify stage completion and zero failed requests."""
    completed_stages = len(
        re.findall(r"Stage\s+\d+\s+-\s+run completed", log_text or "")
    )
    failed_counts = [
        int(m)
        for m in re.findall(r'"?failed_requests"?\s*:\s*(\d+)', log_text or "")
    ]
    failure_block_counts = [
        int(m)
        for m in re.findall(
            r'"?failures"?\s*:\s*\{\s*"?count"?\s*:\s*(\d+)', log_text or ""
        )
    ]
    if len(failure_block_counts) > expected_stages:
        block_failed = max(failure_block_counts)
    else:
        block_failed = sum(failure_block_counts)
    total_failed = sum(failed_counts) + block_failed

    error_lines = []
    for raw_line in (log_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if (
            "prometheus_client" in line
            and "host='localhost', port=9090" in line
        ):
            continue
        if re.search(r"\b(ERROR|CRITICAL|Traceback)\b", line):
            error_lines.append(line)

    passed = (
        completed_stages >= expected_stages
        and total_failed == 0
        and len(error_lines) == 0
    )
    return {
        "passed": passed,
        "completed_stages": completed_stages,
        "expected_stages": expected_stages,
        "failed_requests": total_failed,
        "error_lines": error_lines,
    }


def resolve_autoscaling_target(router_cfg, release_name):
    """Resolves HPA target metadata for either EPP or standalone service-mode proxy."""
    router = (router_cfg or {}).get("router") or {}
    proxy = router.get("proxy") or {}
    proxy_mode = str(proxy.get("mode") or "sidecar").lower()
    proxy_autoscaling = proxy.get("autoscaling") or {}
    epp = router.get("epp") or {}
    epp_autoscaling = epp.get("autoscaling") or {}

    if proxy_mode == "service" and proxy_autoscaling.get("enabled") is True:
        return {
            "component": "proxy",
            "hpa_name": f"{release_name}-proxy",
            "target_label": f"llm-d-router-proxy={release_name}-proxy",
            "epp_label": f"llm-d-router-gateway={release_name}-epp",
            "service_name": f"{release_name}-proxy",
            "target_container": "envoy-proxy",
            "min_replicas": int(proxy_autoscaling.get("minReplicas", 1)),
            "target_cpu_pct": int(
                proxy_autoscaling.get("targetCPUUtilizationPercentage", 80)
            ),
        }

    return {
        "component": "epp",
        "hpa_name": f"{release_name}-epp",
        "target_label": f"llm-d-router-gateway={release_name}-epp",
        "epp_label": f"llm-d-router-gateway={release_name}-epp",
        "service_name": f"{release_name}-epp",
        "target_container": "epp",
        "min_replicas": int(epp_autoscaling.get("minReplicas", 1)),
        "target_cpu_pct": int(
            epp_autoscaling.get("targetCPUUtilizationPercentage", 80)
        ),
    }


def evaluate_autoscaling_run(
    timeline,
    min_replicas=1,
    target_cpu_pct=70,
    epp_log_errors=None,
    perf_audit=None,
    target_container="epp",
):
    """Evaluates whether the autoscaling run satisfied all scale-up, scale-down, CPU, and error criteria."""
    failures = []
    if not timeline:
        return {
            "passed": False,
            "scale_up_verified": False,
            "scale_down_verified": False,
            "epp_cpu_metrics_verified": False,
            "failures": ["No monitoring samples collected."],
        }

    replica_series = [
        max(
            s.get("hpa", {}).get("current_replicas", 0),
            s.get("deployment", {}).get("ready_replicas", 0),
        )
        for s in timeline
    ]
    desired_series = [
        s.get("hpa", {}).get("desired_replicas", 0) for s in timeline
    ]
    hpa_cpu_series = [
        s.get("hpa", {}).get("current_cpu_utilization_pct")
        for s in timeline
        if s.get("hpa", {}).get("current_cpu_utilization_pct") is not None
    ]
    epp_cpu_series = [
        s.get("resources", {})
        .get("containers", {})
        .get(target_container, {})
        .get("cpu_m", 0)
        for s in timeline
    ]
    restarts_series = [
        s.get("deployment", {}).get("restarts", 0) for s in timeline
    ]

    initial_replicas = replica_series[0] if replica_series else 0
    peak_replicas = max(replica_series) if replica_series else 0
    peak_desired = max(desired_series) if desired_series else 0
    peak_idx = replica_series.index(peak_replicas) if replica_series else 0
    post_peak_replicas = replica_series[peak_idx:] if replica_series else []
    min_post_peak_replicas = (
        min(post_peak_replicas) if post_peak_replicas else peak_replicas
    )
    final_replicas = replica_series[-1] if replica_series else 0

    peak_epp_cpu_m = max(epp_cpu_series) if epp_cpu_series else 0
    peak_hpa_cpu_pct = max(hpa_cpu_series) if hpa_cpu_series else 0
    max_restarts = max(restarts_series) if restarts_series else 0

    epp_cpu_metrics_verified = (
        peak_epp_cpu_m > 0
        and len(hpa_cpu_series) > 0
        and peak_hpa_cpu_pct >= target_cpu_pct
    )
    if not epp_cpu_metrics_verified:
        failures.append(
            f"Target ({target_container}) CPU metrics did not exceed HPA target: peak_cpu_m={peak_epp_cpu_m}m, "
            f"peak_hpa_cpu_pct={peak_hpa_cpu_pct}% (target={target_cpu_pct}%)."
        )

    scale_up_verified = peak_replicas > min_replicas and peak_desired > min_replicas
    if not scale_up_verified:
        failures.append(
            f"Scale-up did not occur: min_replicas={min_replicas}, peak_replicas={peak_replicas}, "
            f"peak_desired={peak_desired}."
        )

    scale_down_verified = (
        scale_up_verified and min_post_peak_replicas < peak_replicas
    )
    if not scale_down_verified:
        failures.append(
            f"Scale-down did not occur after peak: peak_replicas={peak_replicas}, "
            f"min_post_peak_replicas={min_post_peak_replicas}, final_replicas={final_replicas}."
        )

    if max_restarts > 0:
        failures.append(f"Detected {max_restarts} container restart(s) in EPP pods.")

    total_epp_errors = sum(len(v) for v in (epp_log_errors or {}).values())
    if total_epp_errors > 0:
        failures.append(
            f"Detected {total_epp_errors} error line(s) in EPP/Envoy logs across pods: "
            f"{list((epp_log_errors or {}).keys())}."
        )

    if perf_audit and not perf_audit.get("passed", False):
        failures.append(
            f"inference-perf audit failed: failed_requests={perf_audit.get('failed_requests')}, "
            f"completed_stages={perf_audit.get('completed_stages')}/{perf_audit.get('expected_stages')}, "
            f"errors={len(perf_audit.get('error_lines', []))}."
        )

    return {
        "passed": len(failures) == 0,
        "scale_up_verified": scale_up_verified,
        "scale_down_verified": scale_down_verified,
        "epp_cpu_metrics_verified": epp_cpu_metrics_verified,
        "initial_replicas": initial_replicas,
        "peak_replicas": peak_replicas,
        "peak_desired_replicas": peak_desired,
        "min_post_peak_replicas": min_post_peak_replicas,
        "final_replicas": final_replicas,
        "peak_epp_cpu_m": peak_epp_cpu_m,
        "peak_hpa_cpu_pct": peak_hpa_cpu_pct,
        "max_restarts": max_restarts,
        "failures": failures,
    }


def sample_hpa(ns, hpa_name):
    res = perf.run_cmd(
        f"kubectl get hpa {hpa_name} -n {ns} -o json", check=False
    )
    if res.returncode != 0 or not res.stdout.strip():
        return {}
    try:
        return parse_hpa_object(json.loads(res.stdout))
    except Exception:
        return {}


def sample_deployment_and_pods(ns, label_selector, pod_logs_cache):
    """Samples deployment status, pod readiness/restarts, and caches logs from all pods."""
    res = perf.run_cmd(
        f"kubectl get pods -n {ns} -l {label_selector} -o json", check=False
    )
    if res.returncode != 0 or not res.stdout.strip():
        return {"ready_replicas": 0, "total_pods": 0, "restarts": 0, "pod_names": []}

    data = json.loads(res.stdout)
    items = data.get("items") or []
    ready_count = 0
    restarts = 0
    pod_names = []

    for pod in items:
        meta = pod.get("metadata") or {}
        pod_name = meta.get("name", "")
        if not pod_name:
            continue
        pod_names.append(pod_name)
        status = pod.get("status") or {}
        for cond in status.get("conditions") or []:
            if cond.get("type") == "Ready" and cond.get("status") == "True":
                ready_count += 1
        for cs in status.get("containerStatuses") or []:
            restarts += int(cs.get("restartCount", 0))
            cname = cs.get("name", "epp")
            log_res = perf.run_cmd(
                f"kubectl logs {pod_name} -n {ns} -c {cname} --tail=500",
                check=False,
            )
            if log_res.returncode == 0 and log_res.stdout:
                pod_logs_cache[f"{pod_name}/{cname}"] = log_res.stdout

    return {
        "ready_replicas": ready_count,
        "total_pods": len(items),
        "restarts": restarts,
        "pod_names": pod_names,
    }


def sample_top_pods(ns, label_selector):
    res = perf.run_cmd(
        f"kubectl top pod -l {label_selector} -n {ns} --containers --no-headers",
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return {
            "pod_count": 0,
            "pods": {},
            "containers": {},
            "total_cpu_m": 0,
            "total_mem_mib": 0,
        }
    return parse_top_containers_output(res.stdout)


def get_hpa_events(ns, hpa_name):
    res = perf.run_cmd(
        f"kubectl get events -n {ns} --field-selector involvedObject.name={hpa_name} "
        f"--sort-by='.lastTimestamp' -o custom-columns=TIME:.lastTimestamp,REASON:.reason,MESSAGE:.message",
        check=False,
    )
    if res.returncode != 0:
        return ""
    return res.stdout.strip()


def scrape_all_epp_scheduler_metrics(ns, label_selector):
    """Aggregates scheduler latency histograms across all active EPP pods."""
    res = perf.run_cmd(
        f"kubectl get pods -n {ns} -l {label_selector} -o jsonpath='{{.items[*].metadata.name}}'",
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return None
    pods = res.stdout.strip().split()
    combined = {"buckets": {}, "sum": 0.0, "count": 0}
    any_success = False
    for pod in pods:
        m = perf.scrape_scheduler_metrics(ns, pod)
        if not m:
            continue
        any_success = True
        combined["sum"] += m.get("sum", 0.0)
        combined["count"] += m.get("count", 0)
        for le, val in (m.get("buckets") or {}).items():
            combined["buckets"][le] = combined["buckets"].get(le, 0.0) + val
    return combined if any_success else None


def write_autoscaling_report(
    report_path,
    ns,
    router_config_path,
    perf_job_path,
    timeline,
    hpa_events,
    verdict,
    p50,
    p95,
    p99,
    perf_audit,
):
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        status_str = "PASS" if verdict["passed"] else "FAIL"
        f.write(
            f"# EPP Autoscaling (HPA) Benchmark & Verification Report ({status_str})\n\n"
        )
        f.write(f"- **Namespace**: `{ns}`\n")
        f.write(f"- **Router Config**: `{router_config_path}`\n")
        f.write(f"- **Workload Config**: `{perf_job_path}`\n")
        f.write(f"- **Overall Verdict**: **{status_str}**\n")
        f.write(
            f"- **Scale-Up Verified**: `{verdict['scale_up_verified']}` "
            f"(Initial: `{verdict['initial_replicas']}` -> Peak: `{verdict['peak_replicas']}` ready / `{verdict['peak_desired_replicas']}` desired)\n"
        )
        f.write(
            f"- **Scale-Down Verified**: `{verdict['scale_down_verified']}` "
            f"(Peak: `{verdict['peak_replicas']}` -> Final: `{verdict['final_replicas']}`)\n"
        )
        f.write(
            f"- **EPP CPU Metrics Verified**: `{verdict['epp_cpu_metrics_verified']}` "
            f"(Peak EPP CPU: `{verdict['peak_epp_cpu_m']}m`, Peak HPA Utilization: `{verdict['peak_hpa_cpu_pct']}%`)\n"
        )
        f.write(
            f"- **Scheduler E2E Latency**: P50 = `{p50:.2f} ms`, P95 = `{p95:.2f} ms`, P99 = `{p99:.2f} ms`\n"
        )
        f.write(f"- **Pod Restarts**: `{verdict['max_restarts']}`\n")
        if perf_audit:
            f.write(
                f"- **Inference-Perf Request Audit**: Completed Stages = `{perf_audit['completed_stages']}/{perf_audit['expected_stages']}`, "
                f"Failed Requests = `{perf_audit['failed_requests']}`\n"
            )
        if verdict["failures"]:
            f.write("\n## Failures\n")
            for fail in verdict["failures"]:
                f.write(f"- {fail}\n")
        if perf_audit and perf_audit.get("error_lines"):
            f.write("\n## Sample Inference-Perf Error Log Lines\n\n```text\n")
            for line in perf_audit["error_lines"][:15]:
                f.write(line + "\n")
            f.write("```\n")

        f.write("\n## HPA & EPP Resource Time-Series\n\n")
        f.write(
            "| Timestamp | HPA Current | HPA Desired | Ready Pods | HPA CPU (%) | Total EPP CPU (m) | Avg EPP CPU/Pod (m) | Total Envoy CPU (m) | Total Pod CPU (m) | Total EPP Mem (MiB) | Per-Pod EPP Breakdown |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for s in timeline:
            ts = s["timestamp"]
            hpa = s.get("hpa") or {}
            dep = s.get("deployment") or {}
            res = s.get("resources") or {}
            epp_c = (res.get("containers") or {}).get("epp") or {}
            env_c = (res.get("containers") or {}).get("envoy-proxy") or {}
            epp_cpu = epp_c.get("cpu_m", 0)
            epp_mem = epp_c.get("mem_mib", 0)
            env_cpu = env_c.get("cpu_m", 0)
            tot_cpu = res.get("total_cpu_m", 0)
            pod_cnt = max(res.get("pod_count", 0), 1)
            avg_epp_cpu = (
                int(round(epp_cpu / pod_cnt)) if res.get("pod_count", 0) > 0 else 0
            )
            hpa_cpu_str = (
                f"{hpa['current_cpu_utilization_pct']}%"
                if hpa.get("current_cpu_utilization_pct") is not None
                else "-"
            )
            per_pod_strs = []
            for p_name, p_ctrs in sorted((res.get("pods") or {}).items()):
                short_name = p_name.split("-")[-1]
                p_epp = (p_ctrs.get("epp") or {}).get("cpu_m", 0)
                p_env = (p_ctrs.get("envoy-proxy") or {}).get("cpu_m", 0)
                per_pod_strs.append(f"`{short_name}`: epp={p_epp}m, envoy={p_env}m")
            per_pod_col = "<br>".join(per_pod_strs) if per_pod_strs else "-"
            f.write(
                f"| {ts} | {hpa.get('current_replicas', '-')} | {hpa.get('desired_replicas', '-')} | "
                f"{dep.get('ready_replicas', '-')} | {hpa_cpu_str} | {epp_cpu} | {avg_epp_cpu} | "
                f"{env_cpu} | {tot_cpu} | {epp_mem} | {per_pod_col} |\n"
            )

        f.write("\n## Kubernetes HPA Events\n\n```text\n")
        f.write((hpa_events or "No HPA events recorded.") + "\n```\n")


def setup_hf_secret_with_fallback(ns):
    if os.environ.get("HF_TOKEN"):
        perf.setup_hf_secret(ns)
        return
    check_sim = perf.run_cmd(
        "kubectl get secret hf-secret -n llm-d-sim", check=False
    )
    if check_sim.returncode == 0:
        perf.setup_hf_secret(ns)
        return
    find_res = perf.run_cmd(
        "kubectl get secrets -A -o jsonpath='{range .items[?(@.metadata.name==\"hf-secret\")]}{.metadata.namespace}{\"\\n\"}{end}'",
        check=False,
    )
    donor_ns = ""
    if find_res.returncode == 0 and find_res.stdout.strip():
        donor_ns = find_res.stdout.strip().splitlines()[0].strip()
    if not donor_ns:
        raise RuntimeError(
            "HF_TOKEN is not set and hf-secret was not found in any cluster namespace."
        )
    print(f"Copying hf-secret from namespace {donor_ns} to {ns}...")
    token_res = perf.run_cmd(
        f"kubectl get secret hf-secret -n {donor_ns} -o jsonpath='{{.data.token}}'"
    )
    token_b64 = token_res.stdout.strip()
    secret_manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "hf-secret", "namespace": ns},
        "type": "Opaque",
        "data": {"token": token_b64},
    }
    tmp_sec = f"/tmp/hf-secret-{ns}.yaml"
    with open(tmp_sec, "w") as f:
        yaml.safe_dump(secret_manifest, f)
    try:
        perf.run_cmd(f"kubectl apply -f {tmp_sec} -n {ns}")
    finally:
        if os.path.exists(tmp_sec):
            os.remove(tmp_sec)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    default_sim_deploy = os.path.join(script_dir, "config", "llm-d-sim-deployment.yaml")
    default_sim_svc = os.path.join(script_dir, "config", "llm-d-sim-service.yaml")
    default_router_chart = os.path.join(
        repo_root, "config", "charts", "llm-d-router-standalone"
    )
    default_perf_chart = os.path.abspath(
        os.path.join(
            script_dir, "..", "..", "..", "..", "inference-perf", "deploy", "inference-perf"
        )
    )
    default_router_config = os.path.join(
        script_dir, "config", "router-configs", "load-aware-session-affinity-hpa.yaml"
    )
    default_perf_job = os.path.join(
        script_dir, "config", "shared_prefix_hpa_scale_up_down.yaml"
    )
    default_report = os.path.join(
        script_dir, "results", "hpa-verification", "epp-autoscaling-report.md"
    )

    parser = argparse.ArgumentParser(
        description="Verify EPP Horizontal Pod Autoscaling (Scale-Up, Scale-Down, CPU Metrics, Zero Errors)"
    )
    parser.add_argument("--namespace", default=None, help="Dedicated namespace name")
    parser.add_argument("--sim-deploy", default=default_sim_deploy)
    parser.add_argument("--sim-svc", default=default_sim_svc)
    parser.add_argument("--router-chart", default=default_router_chart)
    parser.add_argument("--router-chart-version", default="0.0.0")
    parser.add_argument("--router-config", default=default_router_config)
    parser.add_argument("--perf-chart", default=default_perf_chart)
    parser.add_argument("--perf-job", default=default_perf_job)
    parser.add_argument("--report-out", default=default_report)
    parser.add_argument("--sim-replicas", type=int, default=5)
    parser.add_argument("--epp-cpu", default="700m", help="EPP container CPU request")
    parser.add_argument("--epp-cpu-limit", default="4", help="EPP container CPU limit for burst headroom during scale-up")
    parser.add_argument("--epp-memory", default="2Gi", help="EPP container memory request")
    parser.add_argument("--router-machine-family", default="e2")
    parser.add_argument("--gcp-project", default="llm-d-scale")
    parser.add_argument("--poll-interval", type=int, default=10, help="Polling interval in seconds")
    parser.add_argument(
        "--scale-down-timeout",
        type=int,
        default=300,
        help="Additional seconds to wait after benchmark completion for HPA scale-down to minReplicas",
    )
    parser.add_argument("--no-cleanup", action="store_true")
    args = parser.parse_args()

    ns = args.namespace if args.namespace else f"llm-d-hpa-{int(time.time())}"
    release_name = os.path.splitext(os.path.basename(args.router_config))[0]

    with open(args.router_config, "r") as f:
        router_cfg = yaml.safe_load(f) or {}
    target_info = resolve_autoscaling_target(router_cfg, release_name)
    hpa_name = target_info["hpa_name"]
    target_label = target_info["target_label"]
    epp_label = target_info["epp_label"]
    service_name = target_info["service_name"]
    target_container = target_info["target_container"]
    min_replicas = target_info["min_replicas"]
    target_cpu_pct = target_info["target_cpu_pct"]

    with open(args.perf_job, "r") as f:
        job_cfg = yaml.safe_load(f) or {}
    expected_stages = len(
        ((job_cfg.get("config") or {}).get("load") or {}).get("stages") or [1]
    )

    timeline = []
    pod_logs_cache = {}
    stop_poller = False

    def poller_loop():
        while not stop_poller:
            ts = time.strftime("%H:%M:%S")
            hpa_state = sample_hpa(ns, hpa_name)
            dep_state = sample_deployment_and_pods(ns, target_label, pod_logs_cache)
            if target_label != epp_label:
                epp_dep_state = sample_deployment_and_pods(ns, epp_label, pod_logs_cache)
                dep_state["restarts"] = dep_state.get("restarts", 0) + epp_dep_state.get("restarts", 0)
            top_state = sample_top_pods(ns, target_label)
            sample = {
                "timestamp": ts,
                "hpa": hpa_state,
                "deployment": dep_state,
                "resources": top_state,
            }
            timeline.append(sample)
            epp_cpu = (
                (top_state.get("containers") or {}).get("epp") or {}
            ).get("cpu_m", 0)
            env_cpu = (
                (top_state.get("containers") or {}).get("envoy-proxy") or {}
            ).get("cpu_m", 0)
            print(
                f"[{ts}] HPA ({target_info['component']}) replicas={hpa_state.get('current_replicas')}/{hpa_state.get('desired_replicas')} "
                f"(ready={dep_state.get('ready_replicas')}) | "
                f"HPA CPU={hpa_state.get('current_cpu_utilization_pct')}% (target={hpa_state.get('target_cpu_utilization_pct')}%) | "
                f"EPP CPU={epp_cpu}m, Envoy CPU={env_cpu}m, Total CPU={top_state.get('total_cpu_m')}m",
                flush=True,
            )
            time.sleep(args.poll_interval)

    poller_thread = None
    verdict = None
    perf_audit = None
    p50, p95, p99 = 0.0, 0.0, 0.0
    hpa_events = ""

    try:
        perf.create_namespace(ns)
        setup_hf_secret_with_fallback(ns)
        perf.setup_perf_sa(ns, False, args.gcp_project)
        perf.deploy_simulators(ns, args.sim_deploy, args.sim_svc, args.sim_replicas)

        perf.deploy_epp(
            ns,
            args.router_chart,
            args.router_chart_version,
            args.router_config,
            epp_cpu=args.epp_cpu,
            epp_memory=args.epp_memory,
            machine_family=args.router_machine_family,
            epp_replicas=min_replicas,
            epp_cpu_limit=args.epp_cpu_limit,
        )

        print("Waiting for metrics-server to report initial target pod CPU metrics...")
        start_wait = time.time()
        while time.time() - start_wait < 180:
            initial_top = sample_top_pods(ns, target_label)
            if initial_top.get("pod_count", 0) >= min_replicas:
                break
            time.sleep(5)

        metrics_before = scrape_all_epp_scheduler_metrics(ns, epp_label)

        poller_thread = Thread(target=poller_loop, daemon=True)
        poller_thread.start()

        perf.run_benchmark(
            ns, args.perf_job, args.perf_chart, release_name, service_name=service_name
        )

        job_pod_res = perf.run_cmd(
            f"kubectl get pods -n {ns} -l app=inference-perf -o jsonpath='{{.items[0].metadata.name}}'",
            check=False,
        )
        perf_logs = ""
        if job_pod_res.returncode == 0 and job_pod_res.stdout.strip():
            job_pod = job_pod_res.stdout.strip()
            log_res = perf.run_cmd(f"kubectl logs {job_pod} -n {ns}", check=False)
            perf_logs = log_res.stdout or ""
        perf_audit = audit_inference_perf_logs(perf_logs, expected_stages=expected_stages)

        print("Verifying HPA scale-down to minReplicas...")
        cooldown_start = time.time()
        while time.time() - cooldown_start < args.scale_down_timeout:
            if timeline:
                latest_hpa = timeline[-1].get("hpa") or {}
                latest_dep = timeline[-1].get("deployment") or {}
                peak_seen = max(
                    s.get("hpa", {}).get("current_replicas", 0) for s in timeline
                )
                if (
                    peak_seen > min_replicas
                    and latest_hpa.get("current_replicas") == min_replicas
                    and latest_dep.get("ready_replicas") == min_replicas
                ):
                    print(f"Scale-down back to {min_replicas} replica(s) confirmed!")
                    break
            time.sleep(args.poll_interval)

        stop_poller = True
        if poller_thread:
            poller_thread.join(timeout=15)

        metrics_after = scrape_all_epp_scheduler_metrics(ns, epp_label)
        p50, p95, p99 = perf.calculate_percentiles(metrics_before, metrics_after)
        hpa_events = get_hpa_events(ns, hpa_name)

        epp_log_errors = {}
        for key, log_text in pod_logs_cache.items():
            errs = find_epp_log_errors(log_text)
            if errs:
                epp_log_errors[key] = errs

        verdict = evaluate_autoscaling_run(
            timeline,
            min_replicas=min_replicas,
            target_cpu_pct=target_cpu_pct,
            epp_log_errors=epp_log_errors,
            perf_audit=perf_audit,
            target_container=target_container,
        )

    finally:
        stop_poller = True
        if poller_thread and poller_thread.is_alive():
            poller_thread.join(timeout=10)
        if verdict is not None:
            write_autoscaling_report(
                args.report_out,
                ns,
                args.router_config,
                args.perf_job,
                timeline,
                hpa_events,
                verdict,
                p50,
                p95,
                p99,
                perf_audit,
            )
            print(f"Autoscaling verification report written to: {args.report_out}")
        if not args.no_cleanup:
            perf.cleanup_namespace(ns)

    if not verdict or not verdict["passed"]:
        print(
            f"EPP Autoscaling Verification FAILED: {verdict['failures'] if verdict else 'execution error'}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("EPP Autoscaling Verification PASSED!")


if __name__ == "__main__":
    main()
