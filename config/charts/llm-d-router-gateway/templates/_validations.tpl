{{/*
common validations
*/}}
{{- define "llm-d-router.validations.gateway.common" -}}
{{- if or (empty $.Values.router.modelServers) (not $.Values.router.modelServers.matchLabels) }}
{{- fail ".Values.router.modelServers.matchLabels is required" }}
{{- end }}
{{- end -}}
