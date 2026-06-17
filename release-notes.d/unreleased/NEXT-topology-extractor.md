---
pr: PLACEHOLDER
url: https://github.com/llm-d/llm-d-router/pull/PLACEHOLDER
author: elevran
date: 2026-06-17
---
Add `topology-extractor` datalayer plugin: stamps each endpoint with a `Topology` attribute (hostname) at creation time. Hostname is read from the Pod hostname field by default, or from a user-configured pod label via the `hostnameLabel` parameter. When a label is configured but absent, the attribute is not set.
