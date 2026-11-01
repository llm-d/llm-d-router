---
type: Decision
title: "Plugin graduation path across Alpha, Beta, and Stable tiers"
description: "Every new plugin starts at Alpha with no compatibility guarantee, graduates to Beta with two-minor-version compatibility, and reaches Stable with within-major compatibility."
tags: ["plugin-framework-contracts", "configuration-helm-and-compatibility", lifecycle]
status: stable
generated: { by: "okf-agent/gemini-2.5-pro", at: "2026-10-29T00:00:00Z" }
verified: { by: "human:conliu", at: "2026-10-29T00:00:00Z" }
---

# Context

Plugin authors and reviewers needed explicit backward-compatibility contracts tied to plugin maturity tiers so experimental iteration does not lock in premature schemas while mature surfaces stay reliable.

# Tradeoff

Applying uniform deprecation shims to brand-new plugins slows iteration, whereas breaking shipped production plugins without a tier contract disrupts upgrades.

# Outcome

Plugins follow a three-tier graduation path: `Alpha` carries no backward-compatibility guarantee and every new plugin starts in `Alpha`; `Beta` guarantees backward compatibility across two minor versions; `Stable` guarantees backward compatibility within a major version.

Related rule: [See linked review rule](../rules/configuration_helm_and_compatibility.md#provide-a-two-release-deprecation-window-for-shipped-user-facing-surfaces).
