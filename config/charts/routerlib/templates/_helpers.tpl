{{/*
Common labels
*/}}
{{- define "llm-d-router.labels" -}}
app.kubernetes.io/name: {{ include "llm-d-router.name" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
{{- end }}

{{/*
Router deployment name
*/}}
{{- define "llm-d-router.name" -}}
{{- $base := .Release.Name | default "default-pool" | lower | trim | trunc 40 -}}
{{ $base }}-epp
{{- end -}}

{{/*
Cluster RBAC unique name
*/}}
{{- define "llm-d-router.cluster-rbac-name" -}}
{{- $base := .Release.Name | default "default-pool" | lower | trim | trunc 40 }}
{{- $ns := .Release.Namespace | default "default" | lower | trim | trunc 40 }}
{{- printf "%s-%s-epp" $base $ns | quote | trunc 84 }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "llm-d-router.selectorLabels" -}}
{{- if eq .Values.router.inferencePool.create false -}}
{{- /* LOGIC FOR STANDALONE EPP MODE */ -}}
llm-d-router-standalone: {{ include "llm-d-router.name" . }}
{{- else -}}
{{- /* LOGIC FOR PARENT (LLM-D-ROUTER-GATEWAY) MODE */ -}}
llm-d-router-gateway: {{ include "llm-d-router.name" . }}
{{- end -}}
{{- end -}}

{{/*
Mode labels
*/}}
{{- define "llm-d-router.modeLabels" -}}
{{- if eq .Values.router.inferencePool.create false -}}
llm-d.ai/igw-mode: llm-d-router-standalone
{{- else -}}
llm-d.ai/igw-mode: llm-d-router-gateway
{{- end -}}
{{- end -}}

{{/*
Return the monitoring provider name.

If router.monitoring.provider.name is unset/empty, default to
prometheusoperator. For backwards compatibility, provider.name=gke still maps
to gmp when no monitoring provider is explicitly set.
*/}}
{{- define "llm-d-router.monitoring.provider.name" -}}
{{- $monitoring := .Values.router.monitoring | default dict -}}
{{- $mp := index $monitoring "provider" | default dict -}}
{{- $mpName := index $mp "name" | default "" -}}
{{- $gatewayProvider := .Values.provider | default dict -}}
{{- $gatewayProviderName := index $gatewayProvider "name" | default "" -}}
{{- if and (kindIs "string" $mpName) (ne (trim $mpName) "") -}}
{{- $mpName -}}
{{- else if eq (lower $gatewayProviderName) "gke" -}}
gmp
{{- else -}}
prometheusoperator
{{- end -}}
{{- end -}}

{{/*
Return the monitoring provider config object.

When router.monitoring.provider.name is unset/empty, use defaults.
For backwards compatibility, provider.gke.autopilot is still honored when
provider.name=gke and no monitoring provider is explicitly set.
*/}}
{{- define "llm-d-router.monitoring.provider" -}}
{{- $monitoring := .Values.router.monitoring | default dict -}}
{{- $mp := index $monitoring "provider" | default dict -}}
{{- $mpName := include "llm-d-router.monitoring.provider.name" . -}}
{{- $gatewayProvider := .Values.provider | default dict -}}
{{- $gatewayProviderName := index $gatewayProvider "name" | default "" -}}
{{- $resolved := dict "name" $mpName -}}
{{- if eq (lower $mpName) "gmp" -}}
  {{- $gmp := index $mp "gmp" | default dict -}}
  {{- $legacyGke := dict -}}
  {{- if and (eq (lower $gatewayProviderName) "gke") (index $gatewayProvider "gke") -}}
    {{- $legacyGke = index $gatewayProvider "gke" -}}
  {{- end -}}
  {{- $_ := set $resolved "gmp" (mergeOverwrite (deepCopy $legacyGke) (deepCopy $gmp)) -}}
{{- else -}}
  {{- $_ := set $resolved "prometheusoperator" (index $mp "prometheusoperator" | default dict) -}}
{{- end -}}
{{- toYaml $resolved -}}
{{- end -}}

{{/*
Return the standalone proxy type.
*/}}
{{- define "llm-d-router.proxyType" -}}
{{- $proxy := .Values.router.proxy | default dict -}}
{{- default "envoy" ($proxy.proxyType | default "envoy") | lower -}}
{{- end -}}

{{/*
Normalize a scalar, comma-separated string, or list of ports into a
comma-separated numeric string.
*/}}
{{- define "llm-d-router.normalizedPortList" -}}
{{- $path := .path -}}
{{- $value := .value -}}
{{- if empty $value -}}
  {{- fail (printf "%s is required" $path) -}}
{{- end -}}
{{- $rawPorts := list -}}
{{- if kindIs "slice" $value -}}
  {{- $rawPorts = $value -}}
{{- else -}}
  {{- $rawPorts = splitList "," (toString $value) -}}
{{- end -}}
{{- $ports := list -}}
{{- range $raw := $rawPorts -}}
  {{- $rawString := trim (toString $raw) -}}
  {{- if not (regexMatch "^[0-9]+$" $rawString) -}}
    {{- fail (printf "%s must contain only numeric ports, got %q" $path $rawString) -}}
  {{- end -}}
  {{- $port := int $rawString -}}
  {{- if or (lt $port 1) (gt $port 65535) -}}
    {{- fail (printf "%s must contain ports between 1 and 65535, got %d" $path $port) -}}
  {{- end -}}
  {{- $ports = append $ports (toString $port) -}}
{{- end -}}
{{- if eq (len $ports) 0 -}}
  {{- fail (printf "%s must contain at least one port" $path) -}}
{{- end -}}
{{- join "," $ports -}}
{{- end -}}

{{/*
Return the standalone proxy listener port exposed by the EPP Service.
The port is selected by the Service port named "http" so selection is
deterministic even when additional Service ports are configured.
*/}}
{{- define "llm-d-router.standaloneProxyListenerPort" -}}
{{- $servicePorts := .Values.router.extraServicePorts | default list -}}
{{- $found := false -}}
{{- $listenerPort := "" -}}
{{- $targetPort := "" -}}
{{- $hasTargetPort := false -}}
{{- range $index, $servicePort := $servicePorts -}}
  {{- if eq (toString (index $servicePort "name")) "http" -}}
    {{- if $found -}}
      {{- fail ".Values.router.extraServicePorts must contain exactly one port named \"http\" when proxyType=agentgateway" -}}
    {{- end -}}
    {{- $found = true -}}
    {{- if not (hasKey $servicePort "port") -}}
      {{- fail (printf ".Values.router.extraServicePorts[%d].port is required for the port named \"http\"" $index) -}}
    {{- end -}}
    {{- $listenerPort = index $servicePort "port" -}}
    {{- if hasKey $servicePort "targetPort" -}}
      {{- $hasTargetPort = true -}}
      {{- $targetPort = index $servicePort "targetPort" -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- if not $found -}}
  {{- fail ".Values.router.extraServicePorts must contain exactly one port named \"http\" when proxyType=agentgateway" -}}
{{- end -}}
{{- if kindIs "slice" $listenerPort -}}
  {{- fail ".Values.router.extraServicePorts[name=http].port must be a single numeric port" -}}
{{- end -}}
{{- $listenerPortString := trim (toString $listenerPort) -}}
{{- if not (regexMatch "^[0-9]+$" $listenerPortString) -}}
  {{- fail (printf ".Values.router.extraServicePorts[name=http].port must be numeric, got %q" $listenerPortString) -}}
{{- end -}}
{{- $listenerPortNumber := int $listenerPortString -}}
{{- if or (lt $listenerPortNumber 1) (gt $listenerPortNumber 65535) -}}
  {{- fail (printf ".Values.router.extraServicePorts[name=http].port must be between 1 and 65535, got %d" $listenerPortNumber) -}}
{{- end -}}
{{- if $hasTargetPort -}}
  {{- $targetPortString := trim (toString $targetPort) -}}
  {{- if and (ne $targetPortString $listenerPortString) (ne $targetPortString "http") -}}
    {{- fail (printf ".Values.router.extraServicePorts[name=http].targetPort must be omitted, %q, or \"http\" when proxyType=agentgateway, got %q" $listenerPortString $targetPortString) -}}
  {{- end -}}
{{- end -}}
{{- $listenerPortString -}}
{{- end -}}

{{/*
Return the standalone EPP model-server target ports.
*/}}
{{- define "llm-d-router.standaloneEndpointTargetPorts" -}}
{{- $ports := list -}}
{{- range .Values.router.modelServers.targetPorts -}}
{{- $ports = append $ports (toString .number) -}}
{{- end -}}
{{- join "," $ports -}}
{{- end -}}

{{/*
Return the agentgateway standalone logical backend service name.
Derives the name from .Values.router.modelServers.matchLabels.app,
falling back to .Release.Name if not set.
*/}}
{{- define "llm-d-router.agentgateway.logicalBackendName" -}}
{{- $appLabel := "" -}}
{{- if and .Values.router.modelServers .Values.router.modelServers.matchLabels -}}
  {{- $appLabel = index .Values.router.modelServers.matchLabels "app" | default "" -}}
{{- end -}}
{{- if not (empty $appLabel) -}}
  {{- $appLabel -}}
{{- else -}}
  {{- .Release.Name -}}
{{- end -}}
{{- end -}}

{{/*
Return the resolved proxy configuration for the current chart.
Standalone uses proxy presets merged with explicit proxy overrides.
*/}}
{{- define "llm-d-router.proxy" -}}
{{- $proxy := deepCopy (.Values.router.proxy | default dict) -}}
{{- $resolved := $proxy -}}
{{- if hasPrefix "llm-d-router-standalone" .Chart.Name -}}
  {{- $proxyType := include "llm-d-router.proxyType" . -}}
  {{- $presets := index $proxy "presets" | default dict -}}
  {{- $preset := deepCopy ((index $presets $proxyType) | default dict) -}}
  {{- $resolved = mergeOverwrite $preset $proxy -}}
  {{- if eq $proxyType "agentgateway" -}}
    {{- $listenerPort := include "llm-d-router.standaloneProxyListenerPort" . | int -}}
    {{- $ports := index $resolved "ports" | default list -}}
    {{- $resolvedPorts := list (dict "containerPort" $listenerPort "name" "http") -}}
    {{- range $index, $port := $ports -}}
      {{- if gt $index 0 -}}
        {{- $resolvedPorts = append $resolvedPorts $port -}}
      {{- end -}}
    {{- end -}}
    {{- $_ := set $resolved "ports" $resolvedPorts -}}
  {{- end -}}
{{- end -}}
{{- $resolved = omit $resolved "agentgateway" "presets" "proxyType" -}}
{{- toYaml $resolved -}}
{{- end -}}

{{/*
Return the rendered proxy ConfigMap data.
*/}}
{{- define "llm-d-router.proxyConfigMapData" -}}
{{- $proxy := include "llm-d-router.proxy" . | fromYaml | default dict -}}
{{- $configMap := index $proxy "configMap" | default dict -}}
{{- $data := deepCopy ((index $configMap "data") | default dict) -}}
{{- if and (hasPrefix "llm-d-router-standalone" .Chart.Name) (eq (include "llm-d-router.proxyType" .) "agentgateway") -}}
  {{- $generated := dict "config.yaml" (include "llm-d-router.proxy.agentgatewayConfig" .) -}}
  {{- $data = mergeOverwrite $data $generated -}}
{{- end -}}
{{- toYaml $data -}}
{{- end -}}



{{/*
Render the default standalone agentgateway proxy config template.
*/}}
{{- define "llm-d-router.proxy.agentgatewayConfig" -}}
{{- $serviceName := include "llm-d-router.agentgateway.logicalBackendName" . -}}
{{- $serviceNamespace := .Release.Namespace -}}
{{- $servicePorts := splitList "," (include "llm-d-router.standaloneEndpointTargetPorts" .) -}}
{{- $backendPort := index $servicePorts 0 -}}
{{- $listenerPort := include "llm-d-router.standaloneProxyListenerPort" . | int -}}
config:
  statsAddr: "0.0.0.0:15020"
  readinessAddr: "0.0.0.0:15021"
binds:
- port: {{ $listenerPort }}
  listeners:
  - name: default
    protocol: HTTP
    routes:
    - name: standalone-epp
      matches:
      - path:
          pathPrefix: /
      backends:
      - service:
          name: {{ printf "%s/%s" $serviceNamespace $serviceName | quote }}
          port: {{ $backendPort }}
        policies:
          inferenceRouting:
            endpointPicker:
              host: {{ printf "127.0.0.1:%v" (.Values.router.epp.extProcPort | default 9002) | quote }}
            destinationMode: passthrough
services:
- name: {{ $serviceName | quote }}
  namespace: {{ $serviceNamespace | quote }}
  hostname: {{ $serviceName | quote }}
  vips: []
  ports:
    {{- range $servicePort := $servicePorts }}
    {{ $servicePort }}: {{ $servicePort }}
    {{- end }}
{{- end -}}

{{/*
EPP resource validations
*/}}
{{- define "llm-d-router.validations.epp.resources" -}}
{{- if not .Values.router.epp.resources }}
{{- fail ".Values.router.epp.resources is required. EPP is a critical component that must have resource requests set." }}
{{- end }}
{{- if not .Values.router.epp.resources.requests }}
{{- fail ".Values.router.epp.resources.requests is required. EPP is a critical component that must have resource requests set." }}
{{- end }}
{{- $_ := required ".Values.router.epp.resources.requests.cpu is required. EPP is a critical component that must have CPU requests set." .Values.router.epp.resources.requests.cpu }}
{{- $_ := required ".Values.router.epp.resources.requests.memory is required. EPP is a critical component that must have memory requests set." .Values.router.epp.resources.requests.memory }}
{{- end -}}

{{/*
EPP generic validations
*/}}
{{- define "llm-d-router.validations.epp" -}}
{{- include "llm-d-router.validations.deprecations" . }}
{{- include "llm-d-router.validations.epp.resources" . }}
{{- include "llm-d-router.validations.epp.inferenceObjectives" . }}
{{- include "llm-d-router.validations.epp.tokenizer" . }}
{{- end -}}

{{/*
Tokenizer validations: require modelName for the render sidecar's command args.
*/}}
{{- define "llm-d-router.validations.epp.tokenizer" -}}
{{- $tokenizer := .Values.router.tokenizer | default dict }}
{{- if and (dig "enabled" false $tokenizer) (not (dig "modelName" "" $tokenizer)) }}
{{- fail ".Values.router.tokenizer.modelName is required when the tokenizer is enabled." }}
{{- end }}
{{- end -}}

{{/*
EPP inferenceObjectives validations
*/}}
{{- define "llm-d-router.validations.epp.inferenceObjectives" -}}
{{- if and (eq .Values.router.inferencePool.create false) .Values.router.inferenceObjectives }}
{{- fail ".Values.router.inferenceObjectives can only be configured when .Values.router.inferencePool.create is true." }}
{{- end }}
{{- end -}}

{{/*
Deprecation validations
*/}}
{{- define "llm-d-router.validations.deprecations" -}}
{{- if .Values.inferenceExtension }}
{{- fail "Top-level 'inferenceExtension' is deprecated. Please migrate your values to 'router.epp' and other 'router.*' fields." }}
{{- end }}
{{- if .Values.inferencePool }}
{{- fail "Top-level 'inferencePool' is deprecated. Please migrate your values to 'router.modelServers' and 'router.inferencePool'." }}
{{- end }}
{{- if .Values.experimentalHttpRoute }}
  {{- if hasPrefix "llm-d-router-gateway" .Chart.Name }}
    {{- fail "Top-level 'experimentalHttpRoute' is deprecated. Please migrate your values to 'httpRoute'." }}
  {{- else }}
    {{- fail "Top-level 'experimentalHttpRoute' is deprecated and not supported in this chart." }}
  {{- end }}
{{- end }}
{{- if .Values.inferenceObjectives }}
{{- fail "Top-level 'inferenceObjectives' is deprecated. Please migrate your values to 'router.inferenceObjectives'." }}
{{- end }}
{{- end -}}
