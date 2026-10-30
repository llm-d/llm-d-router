{{- define "llm-d-router.crossReplica.passwordEnv" -}}EPP_REDIS_PASSWORD{{- end -}}

{{/* Plugin dependencies are resolved by EPP, including custom plugin types. */}}
{{- define "llm-d-router.crossReplica.notes" -}}
{{- if (.Values.router.epp.crossReplica | default dict).enabled }}
WARNING: Redis configuration alone does not enable shared in-flight load.
The unmodified default and optimized-baseline profiles do not consume in-flight load.
Select payload-agnostic.yaml or configure a consumer such as active-request-scorer
in the scheduling profile. Keep syncCrossReplicaState enabled on its
inflight-load-producer. Approximate prefix-cache state remains local.
{{- end -}}
{{- end -}}

{{/* Augment the selected config without changing its scheduling profile. */}}
{{- define "llm-d-router.crossReplica.config" -}}
{{- $root := .root -}}
{{- $config := .config -}}
{{- $crossReplica := $root.Values.router.epp.crossReplica | default dict -}}
{{- $redis := $crossReplica.redis | default dict -}}
{{- if or (ne $config.apiVersion "llm-d.ai/v1") (ne $config.kind "EndpointPickerConfig") -}}
  {{- fail "router.epp.crossReplica.enabled requires an llm-d.ai/v1 EndpointPickerConfig" -}}
{{- end -}}
{{- if not $redis.address -}}
  {{- fail "router.epp.crossReplica.redis.address is required" -}}
{{- end -}}
{{- if hasKey $redis "password" -}}
  {{- fail "router.epp.crossReplica.redis does not accept a plaintext password; use passwordSecretRef" -}}
{{- end -}}
{{- $parameters := dict "address" $redis.address "db" $redis.db "stateTTL" $redis.stateTTL "coordinationTTL" $redis.coordinationTTL -}}
{{- with $redis.passwordSecretRef -}}
  {{- if or (not .name) (not .key) -}}
    {{- fail "router.epp.crossReplica.redis.passwordSecretRef requires name and key" -}}
  {{- end -}}
  {{- $envName := include "llm-d-router.crossReplica.passwordEnv" $root -}}
  {{- range $root.Values.router.epp.env -}}
    {{- if eq .name $envName -}}
      {{- fail (printf "%s is reserved for router.epp.crossReplica.redis.passwordSecretRef" $envName) -}}
    {{- end -}}
  {{- end -}}
  {{- $_ := set $parameters "passwordEnv" $envName -}}
{{- end -}}
{{- $plugins := $config.plugins | default list -}}
{{- range $plugins -}}
  {{- if or (eq .type "redis-syncer") (eq .name "redis-syncer") -}}
    {{- fail "selected EPP config already configures redis-syncer; configure it manually or use router.epp.crossReplica" -}}
  {{- end -}}
{{- end -}}
{{- $_ := set $config "plugins" (append $plugins (dict "type" "redis-syncer" "parameters" $parameters)) -}}
{{- $dataLayer := $config.dataLayer | default dict -}}
{{- if hasKey $dataLayer "crossReplica" -}}
  {{- fail "selected EPP config already configures crossReplica; configure it manually or use router.epp.crossReplica" -}}
{{- end -}}
{{- $_ := set $dataLayer "crossReplica" (dict "syncerPluginRef" "redis-syncer" "syncInterval" $crossReplica.syncInterval "publishTimeout" $crossReplica.publishTimeout) -}}
{{- $_ := set $config "dataLayer" $dataLayer -}}
{{- toYaml $config -}}
{{- end -}}
