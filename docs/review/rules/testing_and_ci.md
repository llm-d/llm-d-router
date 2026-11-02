# Testing and CI

## Exercise production factories and assert exact behavioral outputs
Instantiate plugins and runners in tests via production factories and `fwkplugin.StrictDecoder` rather than literal internal structs. Assert exact output fields, error types, metric values, and JSON wire payloads without broad `cmpopts.IgnoreFields`, covering both states of feature gates, skipped disaggregated stages, and boundary inputs.
- Accept: End-to-end flow tests over unreachable unit combinations, and calling pure parsers directly in table-driven unit tests separate from env-var wrappers.

## Colocate unit tests and synchronize without fixed sleeps
Keep unit tests in colocated `*_test.go` files moved alongside production types, and place shared mocks or builders in cycle-free leaf test packages. Synchronize async startup and reconciles with injectable clocks, barriers, or `require.Eventually` instead of `time.Sleep`, and bind ephemeral loopback listeners atomically without close-then-rebind races.
- Accept: Calibrated sleep windows when testing refresh-on-read cache TTL expiration where polling the read method would reset the TTL.

## Isolate parallel end-to-end suites with non-aborting teardown
Run E2E suites against freshly built PR images and shipping Helm charts, isolating parallel runs by namespace and port (per-process NodePorts on kind; supervised `kubectl port-forward` on shared clusters). Record created cluster resources before readiness waits, run non-aborting cleanup across all resources on teardown, and dump pod logs only on failure.
- Accept: Separate E2E runner scripts backed by shared shell or Go helpers when cluster ownership, image sets, and environment contracts differ.
