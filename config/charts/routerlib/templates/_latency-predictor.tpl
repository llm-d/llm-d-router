{{/*
Latency predictor deployment mode: "sidecar" (training and prediction servers
run as containers in the EPP pod) or "service" (the training server and the
prediction servers run as their own Deployments, each behind a Service).
*/}}
{{- define "llm-d-router.latencyPredictor.mode" -}}
{{- $lp := .Values.router.latencyPredictor | default dict -}}
{{- default "sidecar" ($lp.mode | default "sidecar") | lower -}}
{{- end -}}

{{/*
Names of the service-mode training and prediction Deployments and Services.
*/}}
{{- define "llm-d-router.latencyPredictor.trainingName" -}}
{{- $base := .Release.Name | default "default-pool" | lower | trim | trunc 40 -}}
{{ $base }}-latency-training
{{- end -}}

{{- define "llm-d-router.latencyPredictor.predictionName" -}}
{{- $base := .Release.Name | default "default-pool" | lower | trim | trunc 40 -}}
{{ $base }}-latency-prediction
{{- end -}}

{{/*
Selector labels for the service-mode Deployments and Services. Distinct from
the EPP selector so the predictor and EPP pods are never co-selected.
*/}}
{{- define "llm-d-router.latencyPredictor.trainingSelectorLabels" -}}
llm-d-router-latency-training: {{ include "llm-d-router.latencyPredictor.trainingName" . }}
{{- end -}}

{{- define "llm-d-router.latencyPredictor.predictionSelectorLabels" -}}
llm-d-router-latency-prediction: {{ include "llm-d-router.latencyPredictor.predictionName" . }}
{{- end -}}

{{/*
Training server URL used by EPP and by the prediction servers for model sync.
Loopback in sidecar mode; the training Service FQDN in service mode.
*/}}
{{- define "llm-d-router.latencyPredictor.trainingURL" -}}
{{- $port := .Values.router.latencyPredictor.trainingServer.port -}}
{{- if eq (include "llm-d-router.latencyPredictor.mode" .) "service" -}}
{{- $domain := .Values.router.clusterDomain | default "cluster.local" -}}
{{- printf "http://%s.%s.svc.%s:%v" (include "llm-d-router.latencyPredictor.trainingName" .) .Release.Namespace $domain $port -}}
{{- else -}}
{{- printf "http://localhost:%v" $port -}}
{{- end -}}
{{- end -}}

{{/*
Comma-separated prediction server URLs for EPP. One loopback URL per sidecar
container in sidecar mode; the prediction Service FQDN in service mode, where
the Service spreads connections across the prediction pods.
*/}}
{{- define "llm-d-router.latencyPredictor.predictionURLs" -}}
{{- $ps := .Values.router.latencyPredictor.predictionServers -}}
{{- $startPort := int $ps.startPort -}}
{{- if eq (include "llm-d-router.latencyPredictor.mode" .) "service" -}}
{{- $domain := .Values.router.clusterDomain | default "cluster.local" -}}
{{- printf "http://%s.%s.svc.%s:%d" (include "llm-d-router.latencyPredictor.predictionName" .) .Release.Namespace $domain $startPort -}}
{{- else -}}
{{- range $i := until (int $ps.count) -}}
{{- if $i }},{{ end }}http://localhost:{{ add $startPort $i }}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Latency Predictor Env
*/}}
{{- define "llm-d-router.latencyPredictor.env" -}}
{{- if .Values.router.latencyPredictor.enabled }}
- name: PREDICTION_SERVER_URL
  value: {{ include "llm-d-router.latencyPredictor.predictionURLs" . | quote }}
- name: TRAINING_SERVER_URL
  value: {{ include "llm-d-router.latencyPredictor.trainingURL" . | quote }}
{{- range $key, $value := .Values.router.latencyPredictor.eppEnv }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Training server container. Rendered in the EPP pod in sidecar mode and in the
training Deployment in service mode.
*/}}
{{- define "llm-d-router.latencyPredictor.trainingContainer" -}}
{{- $ts := .Values.router.latencyPredictor.trainingServer }}
- name: training-server
  image: {{ $ts.image.registry }}/{{ $ts.image.repository }}:{{ $ts.image.tag }}
  imagePullPolicy: {{ $ts.image.pullPolicy }}
  {{- with $ts.securityContext }}
  securityContext:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with $ts.command }}
  command:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with $ts.args }}
  args:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  ports:
  - containerPort: {{ $ts.port }}
    name: training-port
  livenessProbe:
    {{- toYaml $ts.livenessProbe | nindent 4 }}
  readinessProbe:
    {{- toYaml $ts.readinessProbe | nindent 4 }}
  resources:
    {{- toYaml $ts.resources | nindent 4 }}
  envFrom:
  - configMapRef:
      name: {{ include "llm-d-router.name" . }}-latency-predictor-training
  env:
  - name: POD_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: SERVER_TYPE
    value: "training"
  volumeMounts:
  - name: training-server-storage
    mountPath: /models
{{- end }}

{{/*
Prediction server container. Takes a dict with "root" (the chart context) and
"index" (0-based ordinal that selects the listen port and the storage volume).
*/}}
{{- define "llm-d-router.latencyPredictor.predictionContainer" -}}
{{- $root := .root }}
{{- $i := int .index }}
{{- $ps := $root.Values.router.latencyPredictor.predictionServers }}
{{- $port := add (int $ps.startPort) $i }}
{{- $ordinal := add $i 1 }}
- name: prediction-server-{{ $ordinal }}
  image: {{ $ps.image.registry }}/{{ $ps.image.repository }}:{{ $ps.image.tag }}
  imagePullPolicy: {{ $ps.image.pullPolicy }}
  {{- with $ps.securityContext }}
  securityContext:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with $ps.command }}
  command:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with $ps.args }}
  args:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  ports:
  - containerPort: {{ $port }}
    name: predict-port-{{ $ordinal }}
  livenessProbe:
    httpGet:
      path: {{ $ps.livenessProbe.httpGet.path }}
      port: {{ $port }}
    initialDelaySeconds: {{ $ps.livenessProbe.initialDelaySeconds }}
    periodSeconds: {{ $ps.livenessProbe.periodSeconds }}
  readinessProbe:
    httpGet:
      path: {{ $ps.readinessProbe.httpGet.path }}
      port: {{ $port }}
    initialDelaySeconds: {{ $ps.readinessProbe.initialDelaySeconds }}
    periodSeconds: {{ $ps.readinessProbe.periodSeconds }}
    failureThreshold: {{ $ps.readinessProbe.failureThreshold }}
  resources:
    {{- toYaml $ps.resources | nindent 4 }}
  envFrom:
  - configMapRef:
      name: {{ include "llm-d-router.name" $root }}-latency-predictor-prediction
  env:
  - name: PREDICT_PORT
    value: "{{ $port }}"
  - name: POD_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: SERVER_TYPE
    value: "prediction-{{ $ordinal }}"
  - name: TRAINING_SERVER_URL
    value: {{ include "llm-d-router.latencyPredictor.trainingURL" $root | quote }}
  volumeMounts:
  - name: prediction-server-{{ $ordinal }}-storage
    mountPath: /server_models
{{- end }}

{{/*
Latency Predictor Sidecar Containers
*/}}
{{- define "llm-d-router.latencyPredictor.containers" -}}
{{- if and .Values.router.latencyPredictor.enabled (eq (include "llm-d-router.latencyPredictor.mode" .) "sidecar") }}
{{- include "llm-d-router.latencyPredictor.trainingContainer" . }}
{{- range $i := until (int .Values.router.latencyPredictor.predictionServers.count) }}
{{- include "llm-d-router.latencyPredictor.predictionContainer" (dict "root" $ "index" $i) }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Scheduling for the service-mode predictor pods. Each key falls back to the
matching router.epp key when empty, which places the pods where the sidecar
containers run.
*/}}
{{- define "llm-d-router.latencyPredictor.scheduling" -}}
{{- $lp := .Values.router.latencyPredictor -}}
{{- with ($lp.affinity | default .Values.router.epp.affinity) }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with ($lp.tolerations | default .Values.router.epp.tolerations) }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{- define "llm-d-router.latencyPredictor.trainingVolume" }}
- name: training-server-storage
  emptyDir:
    sizeLimit: {{ .Values.router.latencyPredictor.trainingServer.volumeSize }}
{{- end }}

{{/*
Prediction server storage volume. Takes the same dict as predictionContainer.
*/}}
{{- define "llm-d-router.latencyPredictor.predictionVolume" }}
- name: prediction-server-{{ add (int .index) 1 }}-storage
  emptyDir:
    sizeLimit: {{ .root.Values.router.latencyPredictor.predictionServers.volumeSize }}
{{- end }}

{{/*
Latency Predictor Sidecar Volumes
*/}}
{{- define "llm-d-router.latencyPredictor.volumes" -}}
{{- if and .Values.router.latencyPredictor.enabled (eq (include "llm-d-router.latencyPredictor.mode" .) "sidecar") }}
{{- include "llm-d-router.latencyPredictor.trainingVolume" . }}
{{- range $i := until (int .Values.router.latencyPredictor.predictionServers.count) }}
{{- include "llm-d-router.latencyPredictor.predictionVolume" (dict "root" $ "index" $i) }}
{{- end }}
{{- end }}
{{- end }}
