{{/*
Latency Predictor Env
*/}}
{{- define "llm-d-router.latencyPredictor.env" -}}
{{- if .Values.router.latencyPredictor.enabled }}
- name: PREDICTION_SERVER_URL
  value: "{{- $count := int .Values.router.latencyPredictor.predictionServers.count -}}
          {{- $startPort := int .Values.router.latencyPredictor.predictionServers.startPort -}}
          {{- range $i := until $count -}}
            {{- if $i }},{{ end }}http://localhost:{{ add $startPort $i }}
          {{- end }}"
- name: TRAINING_SERVER_URL
  value: "http://localhost:{{ .Values.router.latencyPredictor.trainingServer.port }}"
{{- range $key, $value := .Values.router.latencyPredictor.eppEnv }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Latency Predictor Sidecar Containers
*/}}
{{- define "llm-d-router.latencyPredictor.containers" -}}
{{- if .Values.router.latencyPredictor.enabled }}
# Training Server Sidecar Container
- name: training-server
  image: {{ .Values.router.latencyPredictor.trainingServer.image.registry }}/{{ .Values.router.latencyPredictor.trainingServer.image.repository }}:{{ .Values.router.latencyPredictor.trainingServer.image.tag }}
  imagePullPolicy: {{ .Values.router.latencyPredictor.trainingServer.image.pullPolicy }}
  ports:
  - containerPort: {{ .Values.router.latencyPredictor.trainingServer.port }}
    name: training-port
  livenessProbe:
    {{- toYaml .Values.router.latencyPredictor.trainingServer.livenessProbe | nindent 4 }}
  readinessProbe:
    {{- toYaml .Values.router.latencyPredictor.trainingServer.readinessProbe | nindent 4 }}
  resources:
    {{- toYaml .Values.router.latencyPredictor.trainingServer.resources | nindent 4 }}
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
{{- range $i := until (int .Values.router.latencyPredictor.predictionServers.count) }}
# Prediction Server Sidecar Container {{ add $i 1 }}
- name: prediction-server-{{ add $i 1 }}
  image: {{ $.Values.router.latencyPredictor.predictionServers.image.registry }}/{{ $.Values.router.latencyPredictor.predictionServers.image.repository }}:{{ $.Values.router.latencyPredictor.predictionServers.image.tag }}
  imagePullPolicy: {{ $.Values.router.latencyPredictor.predictionServers.image.pullPolicy }}
  command: ["uvicorn"]
  args: ["llm_d_latency_predictor.prediction_server:app", "--host", "0.0.0.0", "--port", "{{ add $.Values.router.latencyPredictor.predictionServers.startPort $i }}"]
  ports:
  - containerPort: {{ add $.Values.router.latencyPredictor.predictionServers.startPort $i }}
    name: predict-port-{{ add $i 1 }}
  livenessProbe:
    httpGet:
      path: {{ $.Values.router.latencyPredictor.predictionServers.livenessProbe.httpGet.path }}
      port: {{ add $.Values.router.latencyPredictor.predictionServers.startPort $i }}
    initialDelaySeconds: {{ $.Values.router.latencyPredictor.predictionServers.livenessProbe.initialDelaySeconds }}
    periodSeconds: {{ $.Values.router.latencyPredictor.predictionServers.livenessProbe.periodSeconds }}
  readinessProbe:
    httpGet:
      path: {{ $.Values.router.latencyPredictor.predictionServers.readinessProbe.httpGet.path }}
      port: {{ add $.Values.router.latencyPredictor.predictionServers.startPort $i }}
    initialDelaySeconds: {{ $.Values.router.latencyPredictor.predictionServers.readinessProbe.initialDelaySeconds }}
    periodSeconds: {{ $.Values.router.latencyPredictor.predictionServers.readinessProbe.periodSeconds }}
    failureThreshold: {{ $.Values.router.latencyPredictor.predictionServers.readinessProbe.failureThreshold }}
  resources:
    {{- toYaml $.Values.router.latencyPredictor.predictionServers.resources | nindent 4 }}
  envFrom:
  - configMapRef:
      name: {{ include "llm-d-router.name" $ }}-latency-predictor-prediction
  env:
  - name: PREDICT_PORT
    value: "{{ add $.Values.router.latencyPredictor.predictionServers.startPort $i }}"
  - name: POD_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: SERVER_TYPE
    value: "prediction-{{ add $i 1 }}"
  - name: TRAINING_SERVER_URL
    value: "http://localhost:{{ $.Values.router.latencyPredictor.trainingServer.port }}"
  volumeMounts:
  - name: prediction-server-{{ add $i 1 }}-storage
    mountPath: /server_models
{{- end }}
{{- end }}
{{- end }}

{{/*
Latency Predictor Volumes
*/}}
{{- define "llm-d-router.latencyPredictor.volumes" -}}
{{- if .Values.router.latencyPredictor.enabled }}
- name: training-server-storage
  emptyDir: 
    sizeLimit: {{ .Values.router.latencyPredictor.trainingServer.volumeSize }}
{{- range $i := until (int .Values.router.latencyPredictor.predictionServers.count) }}
- name: prediction-server-{{ add $i 1 }}-storage
  emptyDir: 
    sizeLimit: {{ $.Values.router.latencyPredictor.predictionServers.volumeSize }}
{{- end }}
{{- end }}
{{- end }}
