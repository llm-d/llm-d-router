# Round Robin Fairness Policy

**Type:** `round-robin-fairness-policy`

The Round Robin fairness policy selects a queue from a priority band using a simple round-robin strategy. It cycles through active flows one by one, guaranteeing that no single flow can starve others, regardless of its volume, directly addressing the [Noisy Neighbor problem](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.5.0/site-src/guides/flow-control.md#why-flow-control-the-llm-queuing-problem) described in the user guide.

## What it does

1.  **State Management**: It maintains a mutable cursor (`roundRobinCursor`) for each Priority Band it governs, storing the last selected flow key.
2.  **Determinism**: Active queues are ordered deterministically by `FlowKey`.
3.  **Work Conserving**: It skips empty queues and only selects from flows that have pending items.

## Unit of Fairness

**Dispatch Attempts** (Turns). Each active flow is given an equal opportunity to dispatch a request in a cyclic order.

## Inputs consumed

This policy consumes structural data from the flow control system:
*   **Active Queues**: Iterates active queues in the priority band via `PriorityBandAccessor.IterateQueues`.
*   **Queue State**: Inspects the length of queues to skip empty ones.
*   **Cursor State**: Reads and updates the cursor stored on the priority band.

## Configuration

This policy does not require any additional configuration parameters beyond its type registration.

```yaml
fairnessPolicyRef: round-robin-fairness-policy
```

## Trade-offs

*   **Global Ordering Violation**: It breaks strict global ordering (as defined by the `OrderingPolicy`) across flows. An item in a flow might be served before a 'better' item in another flow depending on the cursor position.
*   **Fair Isolation**: Guarantees that no single flow can starve others.

## Related Documentation
*   [Fairness Overview](../README.md)
*   [Flow Control User Guide](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.5.0/site-src/guides/flow-control.md)
