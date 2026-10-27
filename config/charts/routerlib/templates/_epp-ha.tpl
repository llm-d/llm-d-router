{{/* Resolve HA settings once for the EPP arguments and lease RBAC. */}}
{{- define "llm-d-router.epp.flags" -}}
{{- $epp := .Values.router.epp -}}
{{- $flags := deepCopy ($epp.flags | default dict) -}}
{{- range $key := list "ha-enable-leader-election" "allow-experimental-plugins" -}}
  {{- if hasKey $flags $key -}}
    {{- $value := toString (index $flags $key) -}}
    {{- $trueValues := list "1" "t" "T" "true" "TRUE" "True" -}}
    {{- $falseValues := list "0" "f" "F" "false" "FALSE" "False" -}}
    {{- if not (or (has $value $trueValues) (has $value $falseValues)) -}}
      {{- fail (printf "router.epp.flags.%s must be a boolean, got %q" $key $value) -}}
    {{- end -}}
    {{- $_ := set $flags $key (has $value $trueValues) -}}
  {{- end -}}
{{- end -}}
{{- $mode := $epp.haMode | default "" -}}
{{- $crossReplica := $epp.crossReplica | default dict -}}
{{- $autoscaling := $epp.autoscaling | default dict -}}
{{- $gkePB := include "llm-d-router.gkePreferredBackends" . | fromYaml | default dict -}}
{{- $priorityRouting := eq (include "llm-d-router.priorityRouting.enabled" .) "true" -}}
{{- if not (has $mode (list "" "leader-elected" "all-active")) -}}
  {{- fail "router.epp.haMode must be empty, leader-elected, or all-active" -}}
{{- end -}}
{{- if $crossReplica.enabled -}}
  {{- if eq $mode "leader-elected" -}}
    {{- fail "router.epp.crossReplica.enabled requires all-active mode" -}}
  {{- end -}}
  {{- $mode = "all-active" -}}
  {{- if and (hasKey $flags "allow-experimental-plugins") (not (index $flags "allow-experimental-plugins")) -}}
    {{- fail "router.epp.crossReplica.enabled requires allow-experimental-plugins=true" -}}
  {{- end -}}
  {{- $_ := set $flags "allow-experimental-plugins" true -}}
  {{- if or (hasKey $flags "config-file") (hasKey $flags "config-text") -}}
    {{- fail "router.epp.crossReplica.enabled cannot override config-file or config-text; use pluginsConfig or pluginsCustomConfig" -}}
  {{- end -}}
{{- end -}}
{{- if and $mode (or $gkePB.enabled $priorityRouting) -}}
  {{- fail "router.epp.haMode is incompatible with priority routing and GKE preferredBackends" -}}
{{- end -}}
{{- if and (eq $mode "leader-elected") $autoscaling.enabled -}}
  {{- fail "EPP autoscaling requires active-active replicas" -}}
{{- end -}}
{{- if $mode -}}
  {{- $leaderElection := eq $mode "leader-elected" -}}
  {{- if and (hasKey $flags "ha-enable-leader-election") (ne $leaderElection (index $flags "ha-enable-leader-election")) -}}
    {{- fail "router.epp.flags.ha-enable-leader-election conflicts with router.epp.haMode (crossReplica.enabled selects all-active)" -}}
  {{- end -}}
  {{- $_ := set $flags "ha-enable-leader-election" $leaderElection -}}
{{- else if and (not (hasKey $flags "ha-enable-leader-election")) (not $autoscaling.enabled) (gt ($epp.replicas | int) 1) (not $gkePB.enabled) (not $priorityRouting) -}}
  {{- $_ := set $flags "ha-enable-leader-election" true -}}
{{- end -}}
{{- toYaml $flags -}}
{{- end -}}

{{/* A single fixed replica requires explicit active-active mode. */}}
{{- define "llm-d-router.epp.allActive" -}}
{{- $epp := .Values.router.epp -}}
{{- $flags := include "llm-d-router.epp.flags" . | fromYaml -}}
{{- $autoscaling := $epp.autoscaling | default dict -}}
{{- $crossReplica := $epp.crossReplica | default dict -}}
{{- $enabled := or $autoscaling.enabled $crossReplica.enabled (eq ($epp.haMode | default "") "all-active") (gt ($epp.replicas | int) 1) -}}
{{- and $enabled (not (index $flags "ha-enable-leader-election")) -}}
{{- end -}}
