#!/usr/bin/env bash
# Checkpoints for the GKE PoC. Run after infra.sh up/annotate and the manifests
# are applied, with kubectl pointing at the PoC cluster.
set -uo pipefail
NS=default
SVC=glm52-server
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
REQUESTS="${REQUESTS:-20}"
pass=0 fail=0
ok() { echo "PASS: $*"; pass=$((pass + 1)); }
ko() { echo "FAIL: $*"; fail=$((fail + 1)); }

kc() { kubectl -n "${NS}" "$@"; }
# A long-lived client pod: `kubectl run --rm -i` can drop output written before
# the attach completes, which made request counts flaky.
kc get pod curl >/dev/null 2>&1 ||
  kc run curl --image=curlimages/curl:8.11.1 --restart=Never --command -- sleep infinity >/dev/null
kc wait --for=condition=Ready pod/curl --timeout=120s >/dev/null
curl_pod() { kc exec curl -- "$@"; }

echo "== 1. Pod-based view: hostNetwork placeholder pods"
# Dual-stack nodes report one InternalIP per family; hostNetwork pods report the
# node's primary (IPv4) one.
node_ip="$(kubectl get node -l exos-poc/control-node=true -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' | awk '{print $1}')"
pod_ips="$(kubectl -n prediction get pods -l app=glm52 -o jsonpath='{range .items[*]}{.status.podIP}{" "}{end}')"
echo "control node IP: ${node_ip}; model pod IPs: ${pod_ips}"
[[ -n "${pod_ips}" && "$(tr ' ' '\n' <<<"${pod_ips}" | sort -u | grep -v '^$')" == "${node_ip}" ]] &&
  ok "every model pod reports the control node IP" || ko "model pod IPs differ from the control node IP"

echo "== 2. fake-ECP slices"
kc get endpointslice -l "kubernetes.io/service-name=${SVC}" \
  -o custom-columns=NAME:.metadata.name,ADDR:.endpoints[0].addresses[0],READY:.endpoints[0].conditions.ready,TYPE:.addressType
ready_addrs="$(kc get endpointslice -l "kubernetes.io/service-name=${SVC}" -o json |
  python3 -c 'import json,sys; print(" ".join(e["addresses"][0] for s in json.load(sys.stdin)["items"] for e in s["endpoints"] if e.get("conditions",{}).get("ready")))')"
[[ -n "${ready_addrs}" ]] && ok "ready runner addresses: ${ready_addrs}" || ko "no ready runners in slices"

echo "== 3. EPP datastore"
state="$(curl_pod curl -s "http://epp.${NS}.svc:9090/debug/plugins/state")"
python3 -c 'import json,sys; s=json.loads(sys.argv[1])["plugins"]["slice-discovery"]["state"]; print(json.dumps(s))' "${state}" 2>/dev/null &&
  ok "EPP reports discovered endpoints" || ko "EPP state not readable"

echo "== 4. Requests through Envoy -> EPP -> runner"
body="{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in three words.\"}],\"max_tokens\":32}"
codes="$(curl_pod sh -c "for i in \$(seq 1 ${REQUESTS}); do curl -s -o /dev/null -w '%{http_code}\n' -H 'content-type: application/json' -d '${body}' http://envoy.${NS}.svc:8080/v1/chat/completions; done")"
echo "status codes: $(sort <<<"${codes}" | uniq -c | tr '\n' ' ')"
[[ "$(grep -c '^200$' <<<"${codes}")" == "${REQUESTS}" ]] && ok "all ${REQUESTS} requests returned 200" || ko "not all requests returned 200"

echo "== 5. Upstreams chosen by EPP (Envoy access log)"
kc logs -l app=envoy --tail=200 2>/dev/null | grep -o 'upstream=[^ ]*' | sort | uniq -c

echo
echo "passed=${pass} failed=${fail}"
[[ ${fail} -eq 0 ]]
