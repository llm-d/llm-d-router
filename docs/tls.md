# TLS

The EPP, the P/D sidecar and the coordinator each terminate TLS on their own listener:

| Component | Listener | Protocol |
|---|---|---|
| EPP | `ext_proc` server the gateway connects to | gRPC |
| P/D sidecar | data-plane listener in front of the model server | HTTP |
| Coordinator | inference listener | HTTP |

A flag that several components accept is spelled and documented identically in each of
them. Which sources supply its value differs: see
[Configuration file keys](#configuration-file-keys).

## Serving flags

| Flag | Values | Default | EPP | Sidecar | Coordinator | Description |
|---|---|---|---|---|---|---|
| `--secure-serving` | `true` / `false` | `true` | yes | yes | yes | Serve the listener over TLS. |
| `--cert-path` | directory path | empty | yes | yes | yes | Directory with `tls.crt` and `tls.key`. |
| `--tls-min-version` | `VersionTLS12`, `VersionTLS13` | `VersionTLS12` | yes | yes | yes | Minimum TLS version for secure serving. A value below `VersionTLS12` is rejected at startup. |
| `--tls-cipher-suites` | Go `crypto/tls` cipher suite names (comma-separated or repeated) | `crypto/tls` default | yes | yes | yes | Cipher suites for secure serving. Only effective for TLS 1.2 and below; TLS 1.3 cipher suites are not configurable. |
| `--enable-cert-reload` | `true` / `false` | `false` | yes | — | — | Reload the `--cert-path` key pair when it changes. |
| `--secure-proxy` | `true` / `false` | `true` | — | deprecated | — | Deprecated alias for `--secure-serving`. |

## Differences between components

| Behavior | EPP | Sidecar | Coordinator |
|---|---|---|---|
| Reloading the certificate key pair | requires `--enable-cert-reload` | whenever a certificate directory is configured | whenever a certificate directory is configured |
| Accepted cipher suite names | the names `crypto/tls` reports, from `tls.CipherSuites()` and `tls.InsecureCipherSuites()` | same as the EPP | also accepts the legacy spellings `TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305` and `TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305`, which `crypto/tls` spells with a `_SHA256` suffix |
| Rejecting an unknown cipher suite name | `unknown cipher suite "NAME"` | same as the EPP | `Cipher suite NAME not supported or doesn't exist` |

## Metrics endpoint (/metrics)

| Flag | Default | EPP | Sidecar | Coordinator | Description |
|---|---|---|---|---|---|
| `--metrics-cert-dir` | empty | yes | yes | yes | Directory with `tls.crt` and `tls.key` for the metrics endpoint. Empty serves metrics over plain HTTP. |
| `--metrics-client-ca-file` | empty | yes | — | — | PEM CA that requires a verified client certificate on the metrics endpoint. |
| `--metrics-endpoint-auth` | `true` | yes | — | — | Authenticate and authorize requests to the metrics endpoint. |

The metrics listener enforces TLS 1.2 and the `crypto/tls` default cipher suites in all three
components. `--tls-min-version` and `--tls-cipher-suites` apply to the serving listener only.

## Configuration file keys

| Component | Keys | Precedence |
|---|---|---|
| EPP | none; the flags are the only source | — |
| P/D sidecar | `secure-serving`, `cert-path`, `tls-min-version`, `tls-cipher-suites`, `metrics-cert-dir` in the YAML passed to `--configuration-file`. `secure-proxy` is a deprecated alias of `secure-serving`; when both appear, `secure-serving` is used | A flag wins; the file supplies values for flags that were not passed |
| Coordinator | `server.secure_serving`, `server.cert_path`, `server.tls_min_version`, `server.tls_cipher_suites`, `server.metrics_cert_dir` in the file passed to `--config` | The file is the primary source; a flag that was passed overrides it |

## Self-signed certificates

When no certificate directory is configured, the listener uses a generated certificate
with subject `O=llm-d` and no subject alternative names, so callers cannot verify it and
have to skip verification (`curl -k`). It is only suitable for testing. Generation is
logged as `creating self-signed TLS certificate`.

## Startup signal

Each component logs the message `server TLS` at startup, whether or not TLS is enabled.
The entry carries two keys: `tls` is `false` when secure serving is disabled, and
`cert_path` is empty when no certificate directory is configured. The rendering of the
keys depends on the log format.
