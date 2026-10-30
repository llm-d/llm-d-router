---
type: Decision
title: "Once-per-request operational logs use V(logging.DEFAULT)"
description: "Once-per-request operational log entries use logger.V(logging.DEFAULT).Info rather than bare logger.Info."
tags: [observability, logging, "agents-md"]
status: stable
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
verified: { by: "human:conliu", at: "2026-10-29T00:00:00Z" }
---

# Context

Logging guidance across repository documentation needed an explicit verbosity level for routine once-per-request operational signals versus startup or anomalous events.

# Tradeoff

Unqualified `logger.Info(...)` emits at verbosity 0 regardless of operator verbosity flags, whereas `logger.V(logging.DEFAULT).Info(...)` (`DEFAULT = 2` in `pkg/common/observability/logging`) aligns per-request operational logs with the named verbosity scale.

# Outcome

Once-per-request operational signals use `logger.V(logging.DEFAULT).Info(...)`. Per-item loops use `logger.V(logging.DEBUG)` guarded by `v.Enabled()`, and state transitions use `logger.V(logging.TRACE)`.

Related rule: [See linked review rule](../rules/observability.md#guard-per-item-debug-loops-and-cap-stateful-plugin-dumps).
