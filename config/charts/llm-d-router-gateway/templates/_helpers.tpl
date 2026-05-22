{{/*
Create a default fully qualified app name for inferenceGateway.
*/}}
{{- define "llm-d-router-gateway.fullname" -}}
  {{- if .Values.experimentalHttpRoute.inferenceGatewayName -}}
    {{- .Values.experimentalHttpRoute.inferenceGatewayName | trunc 63 | trimSuffix "-" -}}
  {{- else -}}
    {{- printf "%s-inference-gateway" .Release.Name| trunc 63 | trimSuffix "-" -}}
  {{- end -}}
{{- end -}}
