# Endpoints API

The router can serve a read-only list of the endpoints it currently tracks, with their labels, on the metrics/admin server:

```text
GET /v1/endpoints
```

The endpoint is off by default. Enable it with `--enable-endpoints-api`.

It is meant for in-cluster clients that need per-pod addresses and labels without calling the Kubernetes API themselves, for example a model server picking a peer with the same rollout label. The router does not interpret the labels; clients filter on whichever label they need.

The response is built from the router's in-memory datastore, so a request does not reach the Kubernetes API. With Kubernetes discovery the datastore holds only Ready pods that match the InferencePool selector. With other discovery sources (for example file discovery) it lists every endpoint the source registered.

## Response format

```json
{
  "endpoints": [
    {
      "name": "model-server-7b79d8d4c9-abcde",
      "namespace": "default",
      "address": "10.42.0.17",
      "port": "8000",
      "rank": 0,
      "labels": {
        "app.kubernetes.io/name": "model-server",
        "pod-template-hash": "7b79d8d4c9"
      }
    }
  ]
}
```

A pod that serves several InferencePool target ports appears once per port. `name` is the pod name in every entry; `port` and `rank` (the port's index in the pool's target ports) tell the entries apart. Entries are sorted by namespace, name, then rank. An empty pool returns `{"endpoints": []}`.

Until discovery finishes its initial sync (the InferencePool is synced in Kubernetes mode, or the discovery plugin reports ready), the endpoint returns `503` so clients do not read a not-yet-populated list as an empty pool; retry with backoff.

Only `GET` is accepted; other methods return `405`.

## Access control

The endpoint shares the metrics/admin server with `/metrics`, pprof and [`/debug/plugins/state`](plugin_debug.md), so the same exposure rules apply. When metrics endpoint authentication is enabled (`--metrics-endpoint-auth`, on by default), its authentication and authorization filters also cover this endpoint. The check is made against the request path, so a client's service account needs RBAC for `nonResourceURLs: ["/v1/endpoints"]` with verb `get`, in addition to sending its token. In file-discovery mode the metrics/admin server does not apply authentication. Pod labels can carry deployment details, so restrict the metrics port with a NetworkPolicy to the clients that need it.
