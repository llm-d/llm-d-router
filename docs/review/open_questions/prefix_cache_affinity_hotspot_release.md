---
type: Open Question
title: "Prefix-cache affinity decode hotspot release formula"
description: "Whether prefix-cache-affinity-filter releases decode hotspots through a unified ttftEstimate formula by default or behind opt-in configuration."
tags: ["performance-on-the-request-path", "prefix-cache", "unr-09"]
status: draft
sources:
  - id: "pr-3278-ic-6229715180"
    resource: "https://github.com/llm-d/llm-d-router/pull/3278#issuecomment-6229715180"
    author: "human:kaushikmitr"
    last_modified: "2026-10-17T04:37:49Z"
  - id: "pr-3421-ic-6233960957"
    resource: "https://github.com/llm-d/llm-d-router/pull/3421#issuecomment-6233960957"
    author: "human:kaushikmitr"
    last_modified: "2026-10-17T12:37:26Z"
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Question

Whether `prefix-cache-affinity-filter` incorporates queue-wait tokens and uncached prefill cost through a unified `ttftEstimate` setting by default or keeps queue-term hotspot release opt-in.[^pr-3278-ic-6229715180][^pr-3421-ic-6233960957]

# Context

Surfaced in conflict audit `UNR-09`. Related rule: [See linked review rule](../rules/performance_on_the_request_path.md#score-prefix-caches-contiguously-and-break-near-ties-by-load).

# Positions

- **Position A**: Keep `prefix-cache-affinity-filter` focused on prefill savings until a unified `ttftEstimate` design avoids double-counting load scorers.[^pr-3278-ic-6229715180]
- **Position B**: Include queue-wait and active prefill load in sticky affinity break conditions so shared system prefixes do not pile onto a single decode pod.[^pr-3421-ic-6233960957]

[^pr-3278-ic-6229715180]: [#3278](https://github.com/llm-d/llm-d-router/pull/3278#issuecomment-6229715180) (human:kaushikmitr)
[^pr-3421-ic-6233960957]: [#3421](https://github.com/llm-d/llm-d-router/pull/3421#issuecomment-6233960957) (human:kaushikmitr)
