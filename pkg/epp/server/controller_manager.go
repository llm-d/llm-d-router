/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package server

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/common"
)

// NewScheme creates a new runtime.Scheme and registers the types based on the config.
func NewScheme(cfg ControllerConfig) *runtime.Scheme {
	s := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(s))
	utilruntime.Must(v1.Install(s))

	if cfg.startCrdReconcilers {
		if cfg.hasInferenceObjective {
			s.AddKnownTypes(cfg.InferenceObjectiveGV,
				&v1alpha2.InferenceObjective{},
				&v1alpha2.InferenceObjectiveList{},
			)
			metav1.AddToGroupVersion(s, cfg.InferenceObjectiveGV)
		}
		if cfg.hasInferenceModelRewrites {
			s.AddKnownTypes(cfg.InferenceModelRewriteGV,
				&v1alpha2.InferenceModelRewrite{},
				&v1alpha2.InferenceModelRewriteList{},
			)
			metav1.AddToGroupVersion(s, cfg.InferenceModelRewriteGV)
		}
	}
	return s
}

// defaultManagerOptions returns the default options used to create the manager.
func defaultManagerOptions(cfg ControllerConfig, gknn common.GKNN, metricsServerOptions metricsserver.Options, scheme *runtime.Scheme) ctrl.Options {
	opt := ctrl.Options{
		Scheme: scheme,
		Cache: cache.Options{
			ByObject: map[client.Object]cache.ByObject{
				&corev1.Pod{}: {
					Namespaces: map[string]cache.Config{
						gknn.Namespace: {},
					},
				},
			},
		},
		Metrics: metricsServerOptions,
	}
	if cfg.startCrdReconcilers {
		if cfg.hasInferenceObjective {
			opt.Cache.ByObject[&v1alpha2.InferenceObjective{}] = cache.ByObject{Namespaces: map[string]cache.Config{
				gknn.Namespace: {},
			}}
		} else {
			ctrl.Log.WithName("controllerManager").Info("Warning: InferenceObjective GVK does not exist on the server. Skipping its reconciler/cache.")
		}
		if cfg.hasInferenceModelRewrites {
			opt.Cache.ByObject[&v1alpha2.InferenceModelRewrite{}] = cache.ByObject{Namespaces: map[string]cache.Config{
				gknn.Namespace: {},
			}}
		} else {
			ctrl.Log.WithName("controllerManager").Info("Warning: InferenceModelRewrite GVK does not exist on the server. Skipping its reconciler/cache.")
		}

		opt.Cache.ByObject[&v1.InferencePool{}] = cache.ByObject{
			Namespaces: map[string]cache.Config{gknn.Namespace: {FieldSelector: fields.SelectorFromSet(fields.Set{
				"metadata.name": gknn.Name,
			})}},
		}
	}
	return opt
}

// NewDefaultManager creates a new controller manager with default configuration.
// Optional override functions can be passed to customize the manager options (e.g., for testing).
func NewDefaultManager(controllerCfg ControllerConfig, gknn common.GKNN, restConfig *rest.Config, metricsServerOptions metricsserver.Options, leaderElectionEnabled bool, overrides ...func(*ctrl.Options)) (ctrl.Manager, error) {
	scheme := NewScheme(controllerCfg)
	opt := defaultManagerOptions(controllerCfg, gknn, metricsServerOptions, scheme)
	if leaderElectionEnabled {
		opt.LeaderElection = true
		opt.LeaderElectionResourceLock = "leases"
		// The lease name needs to be unique per EPP deployment.
		opt.LeaderElectionID = fmt.Sprintf("epp-%s-%s.llm-d.ai", gknn.Namespace, gknn.Name)
		opt.LeaderElectionNamespace = gknn.Namespace
		opt.LeaderElectionReleaseOnCancel = true
	}

	for _, override := range overrides {
		override(&opt)
	}

	manager, err := ctrl.NewManager(restConfig, opt)

	if err != nil {
		return nil, fmt.Errorf("failed to create controller manager: %v", err)
	}
	return manager, nil
}
