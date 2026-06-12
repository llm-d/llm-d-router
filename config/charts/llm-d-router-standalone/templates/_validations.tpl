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
{{- $failOpen := index $proxy "failOpen" -}}
{{- if and (not (kindIs "invalid" $failOpen)) (not (kindIs "bool" $failOpen)) -}}
  {{- fail (printf ".Values.router.proxy.failOpen must be a boolean, got %q" (toString $failOpen)) -}}
{{- end -}}
{{- if eq $proxyMode "service" -}}
  {{- if not $proxy.enabled -}}
    {{- fail ".Values.router.proxy.enabled must be true when .Values.router.proxy.mode=service" -}}
  {{- end -}}
  {{- $proxyType := default "envoy" ($proxy.proxyType | default "envoy") | lower -}}
  {{- if ne $proxyType "envoy" -}}
    {{- fail (printf ".Values.router.proxy.mode=service currently supports only proxyType=envoy, got %q" $proxyType) -}}
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
      {{- fail ".Values.router.proxy.agentgateway is no longer supported; standalone agentgateway derives its logical backend from router.modelServers settings" -}}
    {{- end -}}
    {{- if ne .Values.router.inferencePool.create false -}}
      {{- fail ".Values.router.inferencePool.create=false is required when proxyType=agentgateway; standalone agentgateway currently supports only service-backed routing" -}}
    {{- end -}}
    {{- $listenerPort := include "llm-d-router.standaloneProxyListenerPort" . -}}
    {{- $flags := .Values.router.epp.flags | default dict -}}
    {{- if and (hasKey $flags "secure-serving") (ne (toString (index $flags "secure-serving")) "false") -}}
      {{- fail ".Values.router.epp.flags.secure-serving must be false when proxyType=agentgateway; standalone agentgateway uses plaintext gRPC to EPP over localhost" -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- end -}}
