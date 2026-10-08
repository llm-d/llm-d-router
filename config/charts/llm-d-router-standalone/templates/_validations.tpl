{{/*
common validations
*/}}
{{- define "llm-d-router.validations.gateway.common" -}}
{{- if ne .Values.router.inferencePool.create false }}
{{- if or (empty $.Values.router.modelServers) (not $.Values.router.modelServers.matchLabels) }}
{{- fail ".Values.router.modelServers.matchLabels is required" }}
{{- end }}
{{- end }}
{{- end -}}

{{/*
standalone validations
*/}}
{{- define "llm-d-router.validations.standalone" -}}
{{- $proxy := .Values.router.proxy | default dict -}}
{{- $proxyMode := include "llm-d-router.proxyMode" . | trim -}}
{{- if not (or (eq $proxyMode "sidecar") (eq $proxyMode "service")) -}}
  {{- fail (printf ".Values.router.proxy.mode must be one of [sidecar, service], got %q" $proxyMode) -}}
{{- end -}}
{{- /* Without an InferencePool the EPP --endpoint-selector is rendered from modelServers.matchLabels; an empty selector is rejected by EPP at startup, so require it here. */ -}}
{{- $useInferencePool := ne .Values.router.inferencePool.create false -}}
{{- if not $useInferencePool -}}
  {{- if or (empty .Values.router.modelServers) (not .Values.router.modelServers.matchLabels) -}}
    {{- fail ".Values.router.modelServers.matchLabels is required when .Values.router.inferencePool.create=false: standalone mode renders the EPP --endpoint-selector from matchLabels and cannot start with an empty selector" -}}
  {{- end -}}
{{- end -}}
{{- $failOpen := index $proxy "failOpen" -}}
{{- if and (not (kindIs "invalid" $failOpen)) (not (kindIs "bool" $failOpen)) -}}
  {{- fail (printf ".Values.router.proxy.failOpen must be a boolean, got %q" (toString $failOpen)) -}}
{{- end -}}
{{- if eq $proxyMode "service" -}}
  {{- if not $proxy.enabled -}}
    {{- fail ".Values.router.proxy.enabled must be true when .Values.router.proxy.mode=service" -}}
  {{- end -}}
  {{- $hasHTTP := false -}}
  {{- range $servicePort := (.Values.router.extraServicePorts | default (list)) -}}
    {{- if eq (toString (index $servicePort "name")) "http" -}}
      {{- $hasHTTP = true -}}
    {{- end -}}
  {{- end -}}
  {{- if not $hasHTTP -}}
    {{- fail ".Values.router.extraServicePorts must contain a port named \"http\" for the proxy listener when .Values.router.proxy.mode=service" -}}
  {{- end -}}
{{- end -}}
{{- if $proxy.enabled -}}
  {{- $proxyType := default "envoy" ($proxy.proxyType | default "envoy") | lower -}}
  {{- if not (or (eq $proxyType "envoy") (eq $proxyType "agentgateway")) -}}
    {{- fail (printf ".Values.router.proxy.proxyType must be one of [envoy, agentgateway], got %q" $proxyType) -}}
  {{- end -}}
  {{- if eq $proxyType "agentgateway" -}}
    {{- if hasKey $proxy "agentgateway" -}}
      {{- fail ".Values.router.proxy.agentgateway is no longer supported; standalone agentgateway uses EPP endpoint discovery with a logical service backend" -}}
    {{- end -}}
    {{- if ne .Values.router.inferencePool.create false -}}
      {{- fail ".Values.router.inferencePool.create=false is required when proxyType=agentgateway; standalone agentgateway currently supports only service-backed routing" -}}
    {{- end -}}
    {{- $listenerPort := include "llm-d-router.standaloneProxyListenerPort" . -}}
    {{- $flags := .Values.router.epp.flags | default dict -}}
    {{- if and (hasKey $flags "secure-serving") (ne (toString (index $flags "secure-serving")) "false") -}}
      {{- fail ".Values.router.epp.flags.secure-serving must be false when proxyType=agentgateway; standalone agentgateway uses plaintext gRPC to EPP (over loopback in sidecar mode, or the EPP Service in service mode)" -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- $autoscaling := dig "autoscaling" dict .Values.router.proxy -}}
{{- if $autoscaling.enabled -}}
  {{- if not $proxy.enabled -}}
    {{- fail "Proxy autoscaling (.Values.router.proxy.autoscaling.enabled=true) requires proxy to be enabled (.Values.router.proxy.enabled=true)" -}}
  {{- end -}}
  {{- if ne $proxyMode "service" -}}
    {{- fail "Proxy autoscaling (.Values.router.proxy.autoscaling.enabled=true) is only supported when proxy mode is set to 'service' (.Values.router.proxy.mode=service)" -}}
  {{- end -}}
  {{- $minReplicas := 1 -}}
  {{- if hasKey $autoscaling "minReplicas" -}}
    {{- $minReplicas = int $autoscaling.minReplicas -}}
    {{- if lt $minReplicas 1 -}}
      {{- fail ".Values.router.proxy.autoscaling.minReplicas must be at least 1" -}}
    {{- end -}}
  {{- end -}}
  {{- $maxReplicas := 5 -}}
  {{- if hasKey $autoscaling "maxReplicas" -}}
    {{- $maxReplicas = int $autoscaling.maxReplicas -}}
    {{- if lt $maxReplicas 1 -}}
      {{- fail ".Values.router.proxy.autoscaling.maxReplicas must be at least 1" -}}
    {{- end -}}
  {{- end -}}
  {{- if lt $maxReplicas $minReplicas -}}
    {{- fail ".Values.router.proxy.autoscaling.maxReplicas must be greater than or equal to minReplicas" -}}
  {{- end -}}
  {{- if and (hasKey $autoscaling "targetCPUUtilizationPercentage") (not (kindIs "invalid" $autoscaling.targetCPUUtilizationPercentage)) -}}
    {{- $cpu := int $autoscaling.targetCPUUtilizationPercentage -}}
    {{- if or (lt $cpu 1) (gt $cpu 100) -}}
      {{- fail ".Values.router.proxy.autoscaling.targetCPUUtilizationPercentage must be between 1 and 100" -}}
    {{- end -}}
  {{- end -}}
  {{- if and (hasKey $autoscaling "targetMemoryUtilizationPercentage") (not (kindIs "invalid" $autoscaling.targetMemoryUtilizationPercentage)) -}}
    {{- $mem := int $autoscaling.targetMemoryUtilizationPercentage -}}
    {{- if or (lt $mem 1) (gt $mem 100) -}}
      {{- fail ".Values.router.proxy.autoscaling.targetMemoryUtilizationPercentage must be between 1 and 100" -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- end -}}
