---
type: Reference
title: Conflict classes and recurring fault lines
description: "Glossary of the five conflict audit classes (79 curated conflicts) and 14 architectural fault lines across llm-d-router reviews."
tags: [conflicts, taxonomy, reference]
status: stable
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
---

# Conflict classes

| Class | Count | Meaning in bundle |
| --- | ---: | --- |
| `MAINTAINER_VS_MAINTAINER` | 24 | Two or more maintainers advocated competing designs; 23 settled outcomes live in `decisions/` and 1 (`MVM-08`) remains in `open_questions/`. |
| `AUTHOR_PUSHBACK_WON` | 20 | Pull request author pushed back with benchmarks or code constraints and reviewers accepted; encoded as `### Reviewers accept` bounds on rules. |
| `NORM_VS_CODE` | 14 | Stated review norm diverged from legacy code paths in the tree; used to scope rules to new or touched code. |
| `NORM_VS_DOCS` | 8 | Stated review norm diverged from existing documentation files; tracked for documentation alignment. |
| `UNRESOLVED` | 13 | Open architectural or policy question without maintainer consensus; tracked in `open_questions/` with `status: draft`. |
