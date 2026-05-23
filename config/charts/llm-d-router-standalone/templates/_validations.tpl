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
{{- if hasKey .Values.router "sidecar" -}}
  {{- fail ".Values.router.sidecar has been refactored to .Values.router.proxy. Please update your values.yaml" -}}
{{- end -}}
{{- $proxy := .Values.router.proxy | default dict -}}
{{- if $proxy.enabled -}}
  {{- $proxyType := default "envoy" ($proxy.proxyType | default "envoy") | lower -}}
  {{- if not (or (eq $proxyType "envoy") (eq $proxyType "agentgateway")) -}}
    {{- fail (printf ".Values.router.proxy.proxyType must be one of [envoy, agentgateway], got %q" $proxyType) -}}
  {{- end -}}
  {{- if eq $proxyType "agentgateway" -}}
    {{- if ne .Values.router.inferencePool.create false -}}
      {{- fail ".Values.router.inferencePool.create=false is required when proxyType=agentgateway; standalone agentgateway currently supports only service-backed routing" -}}
    {{- end -}}
    {{- $agentgateway := index $proxy "agentgateway" | default dict -}}
    {{- $service := index $agentgateway "service" | default dict -}}
    {{- $serviceName := index $service "name" | default "" -}}
    {{- $serviceCreate := index $service "create" | default true -}}
    {{- if hasKey $service "port" -}}
      {{- fail ".Values.router.proxy.agentgateway.service.port has been replaced by .Values.router.proxy.agentgateway.service.ports" -}}
    {{- end -}}
    {{- if empty $serviceName -}}
      {{- fail ".Values.router.proxy.agentgateway.service.name is required when proxyType=agentgateway" -}}
    {{- end -}}
    {{- $targetPorts := include "llm-d-router.standaloneEndpointTargetPorts" . -}}
    {{- $servicePorts := include "llm-d-router.agentgateway.modelServicePorts" . -}}
    {{- if ne $targetPorts $servicePorts -}}
      {{- fail (printf ".Values.router.proxy.agentgateway.service.ports must match .Values.router.modelServers.targetPorts when proxyType=agentgateway, got service ports %q and target ports %q" $servicePorts $targetPorts) -}}
    {{- end -}}
    {{- $listenerPort := include "llm-d-router.standaloneProxyListenerPort" . -}}
    {{- $flags := .Values.router.epp.flags | default dict -}}
    {{- if and (hasKey $flags "secure-serving") (ne (toString (index $flags "secure-serving")) "false") -}}
      {{- fail ".Values.router.epp.flags.secure-serving must be false when proxyType=agentgateway; standalone agentgateway uses plaintext gRPC to EPP over localhost" -}}
    {{- end -}}
    {{- if $serviceCreate -}}
      {{- $selectorLabels := include "llm-d-router.agentgateway.modelServiceSelectorLabels" . -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- end -}}
