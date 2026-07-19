#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import subprocess
import socket
import sys
import time
from threading import Thread
import urllib.request
import yaml

# Global flag to stop the metric monitoring thread
stop_monitoring = False

def run_cmd(cmd, check=True, capture_output=True, text=True):
    print(f"Running: {cmd}")
    if isinstance(cmd, str):
        res = subprocess.run(cmd, shell=True, check=check, capture_output=capture_output, text=text)
    else:
        res = subprocess.run(cmd, check=check, capture_output=capture_output, text=text)
    return res

def create_namespace(ns):
    print(f"Creating namespace: {ns}")
    run_cmd(f"kubectl create namespace {ns}")

def setup_hf_secret(ns):
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Creating hf-secret from HF_TOKEN environment variable...")
        run_cmd(f"kubectl create secret generic hf-secret --from-literal=token={hf_token} -n {ns}")
    else:
        print("HF_TOKEN not set in environment. Falling back to copying hf-secret from llm-d-sim namespace...")
        cmd = f"kubectl get secret hf-secret -n llm-d-sim -o yaml | sed 's/namespace: llm-d-sim/namespace: {ns}/g' | kubectl apply -f -"
        run_cmd(cmd)

def setup_perf_sa(ns, enable_workload_identity=False, gcp_project=None):
    print("Creating service account inference-perf-sa...")
    run_cmd(f"kubectl create serviceaccount inference-perf-sa -n {ns}")
    if enable_workload_identity:
        if not gcp_project:
            raise RuntimeError("GCP Project ID is required to configure Workload Identity.")
        print(f"Annotating service account with Workload Identity for project {gcp_project}...")
        run_cmd(
            f"kubectl annotate serviceaccount inference-perf-sa -n {ns} "
            f"iam.gke.io/gcp-service-account=inference-perf-gsa@{gcp_project}.iam.gserviceaccount.com"
        )

def deploy_simulators(ns, sim_deploy_path, sim_svc_path, replicas=10):
    print(f"Deploying simulators to namespace: {ns} with {replicas} replicas")
    
    # Process deployment yaml
    with open(sim_deploy_path, "r") as f:
        deploy_docs = list(yaml.safe_load_all(f))
        
    for doc in deploy_docs:
        if not doc:
            continue
        doc["metadata"]["namespace"] = ns
        if doc.get("kind") == "Deployment":
            if "spec" in doc:
                doc["spec"]["replicas"] = replicas

    temp_deploy = "/tmp/temp-sim-deploy.yaml"
    with open(temp_deploy, "w") as f:
        yaml.safe_dump_all(deploy_docs, f)

    run_cmd(f"kubectl apply -f {temp_deploy} -n {ns}")

    # Process service yaml
    with open(sim_svc_path, "r") as f:
        svc_docs = list(yaml.safe_load_all(f))
        
    for doc in svc_docs:
        if not doc:
            continue
        doc["metadata"]["namespace"] = ns

    temp_svc = "/tmp/temp-sim-svc.yaml"
    with open(temp_svc, "w") as f:
        yaml.safe_dump_all(svc_docs, f)

    run_cmd(f"kubectl apply -f {temp_svc} -n {ns}")
    
    # Wait for rollout
    print("Waiting for llm-d-sim deployment to roll out...")
    run_cmd(f"kubectl rollout status deployment/llm-d-sim -n {ns} --timeout=10m")

def double_cpu(cpu_str):
    if cpu_str.endswith('m'):
        val = int(cpu_str[:-1])
        return f"{val * 2}m"
    val = float(cpu_str)
    if val.is_integer():
        return str(int(val * 2))
    return str(val * 2)

def double_memory(mem_str):
    import re
    match = re.match(r'^(\d+)([a-zA-Z]+)$', mem_str.strip())
    if match:
        val = int(match.group(1))
        unit = match.group(2)
        return f"{val * 2}{unit}"
    val = int(mem_str)
    return str(val * 2)

def deploy_epp(ns, chart_path, chart_version, router_config_path, epp_cpu="2", epp_memory="4Gi", machine_family=None):
    print(f"Deploying EPP standalone using Helm chart from: {chart_path} (version: {chart_version})")
    
    if not os.path.exists(router_config_path):
        raise FileNotFoundError(f"Router config file not found at: {router_config_path}")
        
    release_name = os.path.splitext(os.path.basename(router_config_path))[0]
    
    epp_registry = os.environ.get("EPP_REGISTRY", "ghcr.io/llm-d")
    epp_repository = os.environ.get("EPP_REPOSITORY", "llm-d-router-endpoint-picker")
    epp_tag = os.environ.get("EPP_TAG", "main")
    
    cpu_limit = double_cpu(epp_cpu)
    mem_limit = double_memory(epp_memory)
    
    overrides = {
        "router": {
            "epp": {
                "image": {
                    "registry": epp_registry,
                    "repository": epp_repository,
                    "tag": epp_tag
                },
                "flags": {
                    "v": 4,
                    "enable-pprof": "true",
                    "tracing": "false"
                },
                "resources": {
                    "requests": {
                        "cpu": epp_cpu,
                        "memory": epp_memory
                    },
                    "limits": {
                        "cpu": cpu_limit,
                        "memory": mem_limit
                    }
                }
            },
            "monitoring": {
                "prometheus": {
                    "auth": {
                        "enabled": False
                    }
                }
            },
            "modelServers": {
                "matchLabels": {
                    "app": "llm-d-sim"
                }
            },
            "proxy": {
                "enabled": True
            }
        }
    }

    # Extract guide specific model server labels and nullify them in overrides to avoid helm deep merge keeping them
    try:
        with open(router_config_path, "r") as f:
            guide_data = yaml.safe_load(f)
            if (guide_data and "router" in guide_data 
                    and "modelServers" in guide_data["router"] 
                    and "matchLabels" in guide_data["router"]["modelServers"]):
                for key in guide_data["router"]["modelServers"]["matchLabels"].keys():
                    if key != "app":
                        overrides["router"]["modelServers"]["matchLabels"][key] = None
    except Exception as e:
        print(f"Warning: Could not parse router config to extract modelServers labels for nullification: {e}")
    
    if machine_family:
        overrides["router"]["epp"]["affinity"] = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "cloud.google.com/machine-family",
                                    "operator": "In",
                                    "values": [machine_family]
                                }
                            ]
                        }
                    ]
                }
            }
        }
        
    temp_overrides_path = f"/tmp/test-overrides-{ns}.yaml"
    with open(temp_overrides_path, "w") as f:
        yaml.safe_dump(overrides, f)
        
    print(f"Generated test-overrides values at: {temp_overrides_path}")
    
    cmd = [
        "helm", "install", release_name, chart_path,
        "-f", router_config_path,
        "-f", temp_overrides_path,
        "-n", ns,
        "--version", chart_version
    ]
    
    run_cmd(cmd)
    
    print("Waiting for EPP deployment to become ready...")
    run_cmd(f"kubectl rollout status deployment/{release_name}-epp -n {ns} --timeout=10m")

def get_epp_pod_name(ns, release_name):
    res = run_cmd(f"kubectl get pods -n {ns} -o jsonpath='{{.items[*].metadata.name}}'")
    pods = res.stdout.strip().split()
    prefix = f"{release_name}-epp"
    for pod in pods:
        if pod.startswith(prefix):
            return pod
    raise Exception(f"EPP pod not found in namespace {ns} starting with prefix: {prefix}. Available pods: {pods}")

def get_pod_containers(ns, pod_name):
    res = run_cmd(
        f"kubectl get pod {pod_name} -n {ns} -o jsonpath='{{.spec.containers[*].name}}'"
    )
    return res.stdout.strip().split()

def get_container_images(ns, pod_name):
    res = run_cmd(
        f"kubectl get pod {pod_name} -n {ns} -o jsonpath='{{.spec.containers[*].image}}'"
    )
    return res.stdout.strip().split()

def sample_resources(ns, pod_name):
    # Output of: kubectl top pod <pod> -n <ns> --containers --no-headers
    # Format: <pod>  <container>  <cpu>  <memory>
    res = run_cmd(f"kubectl top pod {pod_name} -n {ns} --containers --no-headers", check=False)
    if res.returncode != 0:
        return None
        
    metrics = {}
    total_cpu = 0
    total_mem = 0
    
    for line in res.stdout.strip().split('\n'):
        parts = line.split()
        if len(parts) < 4:
            continue
        container = parts[1]
        cpu_str = parts[2]
        mem_str = parts[3]
        
        # Parse CPU (e.g. 100m -> 100, or 2 -> 2000)
        if cpu_str.endswith('m'):
            cpu = int(cpu_str[:-1])
        else:
            cpu = int(float(cpu_str) * 1000)
            
        # Parse Memory (e.g. 500Mi -> 500)
        if mem_str.endswith('Mi'):
            mem = int(mem_str[:-2])
        elif mem_str.endswith('Gi'):
            mem = int(float(mem_str[:-2]) * 1024)
        else:
            mem = int(mem_str)
            
        metrics[container] = {'cpu': cpu, 'mem': mem}
        total_cpu += cpu
        total_mem += mem
        
    metrics['TOTAL'] = {'cpu': total_cpu, 'mem': total_mem}
    return metrics

def monitor_resources_loop(ns, pod_name, interval, peak_metrics):
    global stop_monitoring
    print("Starting background resource monitoring...")
    while not stop_monitoring:
        sample = sample_resources(ns, pod_name)
        if sample:
            for container, usage in sample.items():
                if container not in peak_metrics:
                    peak_metrics[container] = {'cpu': 0, 'mem': 0}
                peak_metrics[container]['cpu'] = max(peak_metrics[container]['cpu'], usage['cpu'])
                peak_metrics[container]['mem'] = max(peak_metrics[container]['mem'], usage['mem'])
        time.sleep(interval)
    print("Stopped background resource monitoring.")

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]

def scrape_scheduler_metrics(ns, pod_name):
    pf_port = find_free_port()
    # We port-forward 9090 from the pod locally in the background, curl, and kill it.
    print(f"Scraping metrics for pod {pod_name} using port-forward on port {pf_port}...")
    pf_process = subprocess.Popen(
        ["kubectl", "port-forward", "-n", ns, f"pod/{pod_name}", f"{pf_port}:9090"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for port-forward to establish
    time.sleep(3)
    
    metrics = {
        'buckets': {},
        'sum': 0.0,
        'count': 0
    }
    
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://localhost:{pf_port}/metrics", timeout=5) as response:
            content = response.read().decode('utf-8')
            
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check primary name
            match = re.match(r'llm_d_epp_scheduler_e2e_duration_seconds_bucket\{le="([^"]+)"\} ([\d.e+-]+)', line)
            if not match:
                match = re.match(r'llm_d_router_epp_scheduler_e2e_duration_seconds_bucket\{le="([^"]+)"\} ([\d.e+-]+)', line)
            if not match:
                match = re.match(r'inference_extension_scheduler_e2e_duration_seconds_bucket\{le="([^"]+)"\} ([\d.e+-]+)', line)
                
            if match:
                le = match.group(1)
                val = float(match.group(2))
                metrics['buckets'][le] = val
                continue
                
            match = re.match(r'llm_d_epp_scheduler_e2e_duration_seconds_sum ([\d.e+-]+)', line)
            if not match:
                match = re.match(r'llm_d_router_epp_scheduler_e2e_duration_seconds_sum ([\d.e+-]+)', line)
            if not match:
                match = re.match(r'inference_extension_scheduler_e2e_duration_seconds_sum ([\d.e+-]+)', line)
            if match:
                metrics['sum'] = float(match.group(1))
                continue
                
            match = re.match(r'llm_d_epp_scheduler_e2e_duration_seconds_count ([\d.e+-]+)', line)
            if not match:
                match = re.match(r'llm_d_router_epp_scheduler_e2e_duration_seconds_count ([\d.e+-]+)', line)
            if not match:
                match = re.match(r'inference_extension_scheduler_e2e_duration_seconds_count ([\d.e+-]+)', line)
            if match:
                metrics['count'] = int(float(match.group(1)))
                continue
    except Exception as e:
        print(f"Error scraping metrics over port-forward: {e}")
        metrics = None
    finally:
        pf_process.terminate()
        pf_process.wait()
        
    return metrics

def interpolate_percentile(sorted_buckets, total_count, percentile):
    target_rank = percentile * total_count
    cumulative_count = 0.0
    prev_le = 0.0
    prev_count = 0.0
    
    for le_str, count in sorted_buckets:
        cumulative_count = count
        if le_str == '+Inf':
            le = float('inf')
        else:
            le = float(le_str)
            
        if cumulative_count >= target_rank:
            if le == float('inf'):
                return prev_le
                
            width = le - prev_le
            if cumulative_count == prev_count:
                return le
                
            fraction = (target_rank - prev_count) / (cumulative_count - prev_count)
            value = prev_le + fraction * width
            return value
            
        prev_le = le
        prev_count = cumulative_count
        
    return prev_le

def calculate_percentiles(before, after):
    if not before or not after:
        return 0.0, 0.0
        
    diff_count = after['count'] - before['count']
    if diff_count <= 0:
        print("No new scheduler events recorded during the test.")
        return 0.0, 0.0
        
    diff_buckets = {}
    for le in after['buckets']:
        if le in before['buckets']:
            diff_buckets[le] = after['buckets'][le] - before['buckets'][le]
        else:
            diff_buckets[le] = after['buckets'][le]
            
    def bucket_key(item):
        le = item[0]
        if le == '+Inf':
            return float('inf')
        return float(le)
        
    sorted_buckets = sorted(diff_buckets.items(), key=bucket_key)
    p50 = interpolate_percentile(sorted_buckets, diff_count, 0.50)
    p95 = interpolate_percentile(sorted_buckets, diff_count, 0.95)
    
    return p50 * 1000, p95 * 1000  # Convert to milliseconds

def run_benchmark(ns, job_values_path, chart_path, release_name):
    print(f"Deploying benchmark job in namespace: {ns}")
    
    # Process job values file to point to EPP local Service URL
    with open(job_values_path, "r") as f:
        job_docs = yaml.safe_load(f)
        
    # Override server url to local namespace service
    job_docs["config"]["server"]["base_url"] = f"http://{release_name}-epp:80"
    job_docs["token"]["hfSecret"]["name"] = "hf-secret"
    job_docs["token"]["hfSecret"]["key"] = "token"
    job_docs["job"]["serviceAccountName"] = "inference-perf-sa"

    temp_job_values = "/tmp/temp-job-values.yaml"
    with open(temp_job_values, "w") as f:
        yaml.safe_dump(job_docs, f)

    run_cmd(f"helm install inference-perf {chart_path} -f {temp_job_values} -n {ns}")
    
    # Wait for the job pod to start and finish
    print("Waiting for inference-perf job to start...")
    job_pod_name = ""
    for _ in range(120):
        res = run_cmd(f"kubectl get pods -n {ns} -l app=inference-perf -o jsonpath='{{.items[0].metadata.name}}'", check=False)
        if res.returncode == 0 and res.stdout.strip():
            job_pod_name = res.stdout.strip()
            break
        time.sleep(5)
        
    if not job_pod_name:
        raise RuntimeError("Timed out waiting for inference-perf job pod to be created.")
        
    print(f"Found benchmark job pod: {job_pod_name}. Waiting for completion...")
    run_cmd(f"kubectl wait --for=condition=complete job/inference-perf-job -n {ns} --timeout=60m")
    print("Benchmark job completed.")

def cleanup_namespace(ns):
    print(f"Cleaning up namespace: {ns}")
    run_cmd(f"kubectl delete namespace {ns} --wait=false")

def write_results_to_markdown_folder(results_dir, test_name, run_time, ns, router_config_path, perf_job, machine_family, sim_replicas, images, idle_metrics, peak_metrics, p50, p95, status, profile_results=None):
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{test_name}.md")
    file_exists = os.path.exists(results_file)
    
    with open(results_file, "a") as f:
        if not file_exists:
            # Write header
            f.write(f"# EPP Router Performance Benchmarking Results: {test_name}\n\n")
            f.write("| Timestamp | Namespace | Router Config | Perf Job | Machine Family | Sim Replicas | EPP Images | Container | Idle CPU (m) | Idle Mem (MiB) | Peak CPU (m) | Peak Mem (MiB) | P50 Latency (ms) | P95 Latency (ms) | CPU Profile | Memory Profile | Status |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            
        epp_images = "<br>".join(images)
        mf_str = machine_family if machine_family else "-"
        
        config_name = os.path.splitext(os.path.basename(router_config_path))[0]
        rel_router_config = os.path.relpath(router_config_path, results_dir)
        rel_perf_job = os.path.relpath(perf_job, results_dir)
        
        config_link = f"[{config_name}]({rel_router_config})"
        job_basename = os.path.basename(perf_job)
        job_link = f"[{job_basename}]({rel_perf_job})"
        
        cpu_link = "-"
        mem_link = "-"
        if profile_results:
            cpu_svg = profile_results.get('cpu_svg')
            cpu_png = profile_results.get('cpu_png')
            cpu_pprof = profile_results.get('cpu_pprof')
            mem_svg = profile_results.get('mem_svg')
            mem_png = profile_results.get('mem_png')
            mem_pprof = profile_results.get('mem_pprof')
            
            if cpu_svg and cpu_pprof:
                rel_cpu_svg = os.path.relpath(cpu_svg, results_dir)
                rel_cpu_pprof = os.path.relpath(cpu_pprof, results_dir)
                if cpu_png:
                    rel_cpu_png = os.path.relpath(cpu_png, results_dir)
                    cpu_link = f"[PNG]({rel_cpu_png}) ([SVG]({rel_cpu_svg}) / [pprof]({rel_cpu_pprof}))"
                else:
                    cpu_link = f"[SVG]({rel_cpu_svg}) ([pprof]({rel_cpu_pprof}))"
                
            if mem_svg and mem_pprof:
                rel_mem_svg = os.path.relpath(mem_svg, results_dir)
                rel_mem_pprof = os.path.relpath(mem_pprof, results_dir)
                if mem_png:
                    rel_mem_png = os.path.relpath(mem_png, results_dir)
                    mem_link = f"[PNG]({rel_mem_png}) ([SVG]({rel_mem_svg}) / [pprof]({rel_mem_pprof}))"
                else:
                    mem_link = f"[SVG]({rel_mem_svg}) ([pprof]({rel_mem_pprof}))"

        # Write rows for each container
        for container in sorted(peak_metrics.keys()):
            idle_cpu = idle_metrics.get(container, {}).get('cpu', '-')
            idle_mem = idle_metrics.get(container, {}).get('mem', '-')
            peak_cpu = peak_metrics.get(container, {}).get('cpu', '-')
            peak_mem = peak_metrics.get(container, {}).get('mem', '-')
            
            f.write(f"| {run_time} | {ns} | {config_link} | {job_link} | {mf_str} | {sim_replicas} | {epp_images} | {container} | {idle_cpu} | {idle_mem} | {peak_cpu} | {peak_mem} | {p50:.2f} | {p95:.2f} | {cpu_link} | {mem_link} | {status} |\n")


def collect_profiles(ns, epp_pod_name, results_dir, test_name, run_time_clean, profile_results):
    os.makedirs(results_dir, exist_ok=True)
    print("Profile collector: waiting for benchmark job to be created...")
    job_pod_name = ""
    for _ in range(120):
        res = run_cmd(f"kubectl get pods -n {ns} -l app=inference-perf -o jsonpath='{{.items[0].metadata.name}}'", check=False)
        if res.returncode == 0 and res.stdout.strip():
            job_pod_name = res.stdout.strip()
            break
        time.sleep(5)
        
    if not job_pod_name:
        print("Profile collector: timed out waiting for benchmark job pod. Profiles will not be collected.")
        return

    print(f"Profile collector: found benchmark pod {job_pod_name}. Waiting for it to start running...")
    pod_running = False
    for _ in range(120): # Wait up to 10 minutes
        phase_res = run_cmd(f"kubectl get pod {job_pod_name} -n {ns} -o jsonpath='{{.status.phase}}'", check=False)
        if phase_res.returncode == 0 and phase_res.stdout.strip() == "Running":
            pod_running = True
            break
        time.sleep(5)
        
    if not pod_running:
        print(f"Profile collector: pod {job_pod_name} did not enter Running state. Exiting.")
        return

    print(f"Profile collector: pod {job_pod_name} is running. Monitoring logs for 'Stage 1 - run started'...")
    
    log_proc = subprocess.Popen(
        ["kubectl", "logs", "-n", ns, job_pod_name, "-f"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    found_started = False
    start_time = time.time()
    timeout = 300
    
    while True:
        if time.time() - start_time > timeout:
            print("Profile collector: timed out waiting for log line 'Stage 1 - run started'")
            break
            
        if log_proc.poll() is not None:
            break
            
        line = log_proc.stdout.readline()
        if not line:
            time.sleep(1)
            continue
            
        if "Stage 1 - run started" in line:
            print("Profile collector: detected 'Stage 1 - run started' in logs!")
            found_started = True
            break
            
    log_proc.terminate()
    log_proc.wait()
    
    if not found_started:
        print("Profile collector: failed to detect run start. Exiting.")
        return
        
    print("Profile collector: waiting 60 seconds before initiating profiling...")
    time.sleep(60)
    
    pf_port = random.randint(30000, 40000)
    print(f"Profile collector: port-forwarding to EPP pod {epp_pod_name} on port {pf_port}...")
    pf_process = subprocess.Popen(
        ["kubectl", "port-forward", "-n", ns, f"pod/{epp_pod_name}", f"{pf_port}:9090"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)
    
    try:
        artifact_dir = os.path.join(results_dir, f"{test_name}-{run_time_clean}")
        os.makedirs(artifact_dir, exist_ok=True)

        cpu_profile_path = os.path.join(artifact_dir, f"{test_name}-{run_time_clean}-cpu.pprof")
        cpu_svg_path = os.path.join(artifact_dir, f"{test_name}-{run_time_clean}-cpu.svg")
        cpu_png_path = os.path.join(artifact_dir, f"{test_name}-{run_time_clean}-cpu.png")
        print("Profile collector: fetching CPU profile (30s)...")
        req = urllib.request.Request(f"http://localhost:{pf_port}/debug/pprof/profile?seconds=30")
        with urllib.request.urlopen(req, timeout=45) as response:
            cpu_data = response.read()
        with open(cpu_profile_path, "wb") as f:
            f.write(cpu_data)
        print(f"Profile collector: CPU profile saved to {cpu_profile_path}")

        mem_profile_path = os.path.join(artifact_dir, f"{test_name}-{run_time_clean}-memory.pprof")
        mem_svg_path = os.path.join(artifact_dir, f"{test_name}-{run_time_clean}-memory.svg")
        mem_png_path = os.path.join(artifact_dir, f"{test_name}-{run_time_clean}-memory.png")
        print("Profile collector: fetching memory (heap) profile...")
        req = urllib.request.Request(f"http://localhost:{pf_port}/debug/pprof/heap")
        with urllib.request.urlopen(req, timeout=15) as response:
            mem_data = response.read()
        with open(mem_profile_path, "wb") as f:
            f.write(mem_data)
        print(f"Profile collector: Memory profile saved to {mem_profile_path}")

        print("Profile collector: generating visualizations using go tool pprof...")
        run_cmd(f"go tool pprof -svg -output {cpu_svg_path} {cpu_profile_path}")
        run_cmd(f"go tool pprof -png -output {cpu_png_path} {cpu_profile_path}")
        run_cmd(f"go tool pprof -svg -output {mem_svg_path} {mem_profile_path}")
        run_cmd(f"go tool pprof -png -output {mem_png_path} {mem_profile_path}")
        print(f"Profile collector: visualizations generated in {artifact_dir}")
        
        profile_results['cpu_pprof'] = cpu_profile_path
        profile_results['cpu_svg'] = cpu_svg_path
        profile_results['cpu_png'] = cpu_png_path
        profile_results['mem_pprof'] = mem_profile_path
        profile_results['mem_svg'] = mem_svg_path
        profile_results['mem_png'] = mem_png_path

    except Exception as e:
        print(f"Profile collector failed with error: {e}")
    finally:
        pf_process.terminate()
        pf_process.wait()



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # Resolve relative defaults
    default_sim_deploy = os.path.join(script_dir, "config", "llm-d-sim-deployment.yaml")
    default_sim_svc = os.path.join(script_dir, "config", "llm-d-sim-service.yaml")
    default_router_chart = "oci://ghcr.io/llm-d/charts/llm-d-router-standalone"
    default_perf_chart = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", "inference-perf", "deploy", "inference-perf"))
    default_perf_job = os.path.join(script_dir, "config", "shared_prefix_job1.yaml")
    default_results_dir = os.path.abspath(os.path.join(script_dir, "results"))
    default_router_config = os.path.abspath(os.path.join(script_dir, "config", "router-configs", "optimized-baseline.yaml"))
 
    parser = argparse.ArgumentParser(description="Nightly EPP Performance Benchmarking")
    parser.add_argument("--namespace", default=None, help="Dedicated namespace name (auto-generated if omitted)")
    parser.add_argument("--sim-deploy", default=default_sim_deploy, help="Path to simulator deployment yaml")
    parser.add_argument("--sim-svc", default=default_sim_svc, help="Path to simulator service yaml")
    parser.add_argument("--router-chart", default=default_router_chart, help="Path to EPP Helm chart")
    parser.add_argument("--router-chart-version", default="v0", help="EPP Helm chart version")
    parser.add_argument("--router-config", default=default_router_config, help="Path to consolidated router Helm values configuration")
    parser.add_argument("--test-name", default="optimized-baseline-job1", help="Name of the performance test")
    parser.add_argument("--perf-chart", default=default_perf_chart, help="Path to inference-perf Helm chart")
    parser.add_argument("--perf-job", default=default_perf_job, help="Path to inference-perf job configuration")
    parser.add_argument("--results-dir", default=default_results_dir, help="Directory to output markdown results")
    parser.add_argument("--sim-replicas", type=int, default=10, help="Number of simulator replicas")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip namespace deletion on completion")
    parser.add_argument("--enable-workload-identity", action="store_true", help="Annotate service account with Workload Identity configuration")
    parser.add_argument("--gcp-project", default=None, help="Google Cloud project ID (inferred from active gcloud config if omitted)")
    parser.add_argument("--epp-cpu", default="2", help="EPP CPU request (limit will be 2x this amount)")
    parser.add_argument("--epp-memory", default="4Gi", help="EPP memory request (limit will be 2x this amount)")
    parser.add_argument("--router-machine-family", default=None, help="Add node affinity for specific Google Cloud machine-family (e.g. c3)")
    parser.add_argument("--collect-pprof-profiles", action="store_true", help="Collect EPP CPU & memory pprof profiles and render them as SVGs")
    parser.add_argument("--gcs-bucket", default=None, help="Optional GCS bucket name or URI (e.g. gs://my-bucket) to sync performance results to")
    args = parser.parse_args()


    global stop_monitoring
    run_time = time.strftime("%Y-%m-%d %H:%M:%S")
    ns = args.namespace if args.namespace else f"llm-d-perf-{int(time.time())}"
    
    print(f"=== EPP Benchmarking Suite Started at {run_time} ===")
    print(f"Namespace: {ns}")

    status = "SUCCESS"
    idle_metrics = {}
    peak_metrics = {}
    p50, p95 = 0.0, 0.0
    images = []
    profile_results = {
        'cpu_pprof': None,
        'cpu_svg': None,
        'cpu_png': None,
        'mem_pprof': None,
        'mem_svg': None,
        'mem_png': None
    }

    try:
        # Step 1: Create Namespace
        create_namespace(ns)

        # Step 2: Setup HF Secret
        setup_hf_secret(ns)

        # Resolve GCP project if needed
        gcp_project = args.gcp_project
        if not gcp_project:
            res = run_cmd("gcloud config get-value project", check=False)
            if res.returncode == 0:
                gcp_project = res.stdout.strip()
            if not gcp_project:
                raise RuntimeError("GCP project ID could not be determined. Please specify it via --gcp-project.")
        print(f"Using GCP Project: {gcp_project}")

        # Step 2.5: Setup inference-perf ServiceAccount
        setup_perf_sa(ns, args.enable_workload_identity, gcp_project)

        # Step 3: Deploy Simulators
        deploy_simulators(ns, args.sim_deploy, args.sim_svc, args.sim_replicas)

        release_name = os.path.splitext(os.path.basename(args.router_config))[0]

        # Step 4: Deploy EPP Standalone Router
        deploy_epp(
            ns, 
            args.router_chart, 
            args.router_chart_version,
            args.router_config,
            epp_cpu=args.epp_cpu, 
            epp_memory=args.epp_memory, 
            machine_family=args.router_machine_family
        )

        # Get EPP pod details
        pod_name = get_epp_pod_name(ns, release_name)
        print(f"EPP Pod name: {pod_name}")
        images = get_container_images(ns, pod_name)
        print(f"EPP Container images: {images}")

        # Step 5: Measure Idle Performance
        print("Measuring idle resource usage (waiting up to 5m for metrics-server)...")
        time.sleep(15)
        idle_samples = []
        start_time = time.time()
        timeout = 300  # 5 minutes
        backoff = 5
        
        while len(idle_samples) < 3:
            if time.time() - start_time > timeout:
                print("Timeout waiting for idle resource metrics.")
                break
                
            sample = sample_resources(ns, pod_name)
            if sample:
                idle_samples.append(sample)
                print(f"Captured idle metrics sample {len(idle_samples)}/3.")
                backoff = 5
                if len(idle_samples) < 3:
                    time.sleep(5)
            else:
                print(f"Metrics not available yet. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        if idle_samples:
            for container in idle_samples[0].keys():
                cpus = [s[container]['cpu'] for s in idle_samples if container in s]
                mems = [s[container]['mem'] for s in idle_samples if container in s]
                idle_metrics[container] = {
                    'cpu': int(sum(cpus)/len(cpus)) if cpus else 0,
                    'mem': int(sum(mems)/len(mems)) if mems else 0
                }
            print(f"Idle metrics captured: {idle_metrics}")
        else:
            print("Warning: failed to capture idle metrics.")

        # Step 6: Scrape Pre-Benchmark Metrics
        metrics_before = scrape_scheduler_metrics(ns, pod_name)

        # Step 7: Execute Benchmark & Monitor Peak Usage
        monitoring_thread = Thread(target=monitor_resources_loop, args=(ns, pod_name, 5, peak_metrics))
        monitoring_thread.start()

        profile_thread = None
        if args.collect_pprof_profiles:
            run_time_clean = run_time.replace(" ", "_").replace(":", "")
            profile_thread = Thread(target=collect_profiles, args=(ns, pod_name, args.results_dir, args.test_name, run_time_clean, profile_results))
            profile_thread.start()

        try:
            run_benchmark(ns, args.perf_job, args.perf_chart, release_name)
        finally:
            stop_monitoring = True
            monitoring_thread.join()
            if profile_thread:
                profile_thread.join()

        # Step 8: Scrape Post-Benchmark Metrics & Compute Latency
        metrics_after = scrape_scheduler_metrics(ns, pod_name)
        p50, p95 = calculate_percentiles(metrics_before, metrics_after)
        print(f"Benchmark latencies: P50 = {p50:.2f} ms, P95 = {p95:.2f} ms")
        print(f"Peak resource usage: {peak_metrics}")

    except Exception as e:
        print(f"Test run failed with exception: {e}", file=sys.stderr)
        status = "FAILED"
        import traceback
        traceback.print_exc()
    finally:
        # Step 9: Teardown & Cleanup
        if not args.no_cleanup:
            cleanup_namespace(ns)
        else:
            print(f"Skipping namespace cleanup. Namespace '{ns}' remains active.")

        # Step 10: Log Results to Markdown File and optional GCS sync
        if idle_metrics or peak_metrics:
            gcs_dest = None
            if args.gcs_bucket:
                bucket = args.gcs_bucket if args.gcs_bucket.startswith("gs://") else f"gs://{args.gcs_bucket}"
                recipe_folder = os.path.basename(os.path.normpath(args.results_dir))
                gcs_dest = f"{bucket}/{recipe_folder}"
                print(f"Syncing existing performance results from GCS bucket {gcs_dest}...")
                run_cmd(f"gcloud storage cp -r {gcs_dest}/* {args.results_dir}/", check=False)

            write_results_to_markdown_folder(
                args.results_dir, 
                args.test_name,
                run_time, 
                ns, 
                args.router_config,
                args.perf_job,
                args.router_machine_family,
                args.sim_replicas,
                images, 
                idle_metrics, 
                peak_metrics, 
                p50, 
                p95, 
                status,
                profile_results=profile_results
            )
            print(f"Performance results successfully written to {args.results_dir}/{args.test_name}.md")

            if gcs_dest:
                print(f"Uploading performance results to GCS bucket {gcs_dest}...")
                run_cmd(f"gcloud storage cp -r {args.results_dir}/* {gcs_dest}/")
                print(f"Successfully uploaded performance results to {gcs_dest}")

if __name__ == "__main__":
    main()
