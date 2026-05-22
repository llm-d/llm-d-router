{{/*
common validations
*/}}
{{- define "llm-d-router.validations.gateway.common" -}}
{{- if or (empty $.Values.inferenceExtension.modelServers) (not $.Values.inferenceExtension.modelServers.matchLabels) }}
{{- fail ".Values.inferenceExtension.modelServers.matchLabels is required" }}
{{- end }}
{{- end -}}
