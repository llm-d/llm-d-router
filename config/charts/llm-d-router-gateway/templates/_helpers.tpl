{{/*
Create a default fully qualified app name for inferenceGateway.
*/}}
{{- define "llm-d-router-gateway.fullname" -}}
  {{- if .Values.httpRoute.inferenceGatewayName -}}
    {{- .Values.httpRoute.inferenceGatewayName | trunc 63 | trimSuffix "-" -}}
  {{- else -}}
    {{- printf "%s-inference-gateway" .Release.Name| trunc 63 | trimSuffix "-" -}}
  {{- end -}}
{{- end -}}
