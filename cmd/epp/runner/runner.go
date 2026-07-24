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

package runner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/spf13/pflag"
	"google.golang.org/grpc"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	"github.com/llm-d/llm-d-router/internal/runnable"
	"github.com/llm-d/llm-d-router/pkg/common"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/profiling"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/epp/config"
	"github.com/llm-d/llm-d-router/pkg/epp/config/loader"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	fccontroller "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	fcregistry "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrgpu "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/gpu"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	discoveryfile "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/discovery/file"
	extdcgm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/dcgm"
	extractormetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	extmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/models"
	exttopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/topology"
	srcdcgm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/dcgm"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	srcmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/models"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	evictfiltering "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/eviction/filtering"
	evictordering "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/eviction/ordering"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	programaware "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/program-aware"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/roundrobin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/edf"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	slodeadline "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/slodeadline"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits/priorityholdback"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits/softreflectiveceiling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/admitter/latencyslo"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/admitter/probabilisticadmitter"
	reqdataprodprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/burstprefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	mmproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/multimodal"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/p2psource"
	preciseproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache"
	latencyproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionid"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestattributereporter"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
	testresponsereceived "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/test/responsereceived"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/passthrough"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vertexai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmhttp"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	endpointattributefilter "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/endpointattribute"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/prefixcacheaffinity"
	sessionaffinityfilter "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/sloheadroomtier"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/random"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/weightedrandom"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/dataparallel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/activerequest"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/contextlengthaware"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/endpointattribute"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization"
	latencyscorer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/latency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loadaware"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loraaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/mmcacheaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/nohitlru"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/preciseprefixcache"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/runningrequests"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/tokenload"
	testfilter "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/test/filter"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics/collectors"
	"github.com/llm-d/llm-d-router/pkg/epp/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
	"github.com/llm-d/llm-d-router/pkg/epp/util/env"
	"github.com/llm-d/llm-d-router/version"
)

const (
	// enableExperimentalFlowControlLayer defines the environment variable used as a feature flag for the pluggable flow
	// control layer.
	// DEPRECATION NOTICE - this env var will be removed in the next version as we switch to configuring the EPP using FeatureGates in the config file.
	enableExperimentalFlowControlLayer = "ENABLE_EXPERIMENTAL_FLOW_CONTROL_LAYER"
)

var (
	setupLog = ctrl.Log.WithName("setup")
)

// NewRunner initializes a new EPP Runner and returns its pointer.
func NewRunner() *Runner {
	return &Runner{
		eppExecutableName:    "GIE",
		requestControlConfig: requestcontrol.NewConfig(), // default requestcontrol config has empty plugin list
		customCollectors:     []prometheus.Collector{},
	}
}

// Runner is used to run epp with its plugins
type Runner struct {
	eppExecutableName    string // the EPP executable name
	featureGates         map[string]bool
	requestControlConfig *requestcontrol.Config
	schedulerConfig      *scheduling.SchedulerConfig
	customCollectors     []prometheus.Collector
	parserRegistry       *handlers.ParserRegistry
	dlRuntime            *datalayer.Runtime
	PluginHandle         fwkplugin.Handle
	// rawConfig caches the result of parseConfigurationPhaseOne.
	rawConfig *configapi.EndpointPickerConfig

	// Populated by setup(); see runWithGracefulShutdown.
	serverRunner     *runserver.ExtProcServerRunner
	healthGRPCServer *grpc.Server
	healthGRPCPort   int
	draining         *atomic.Bool
}

// WithExecutableName sets the name of the executable containing the runner.
// The name is used in the version log upon startup and is otherwise opaque.
func (r *Runner) WithExecutableName(exeName string) *Runner {
	r.eppExecutableName = exeName
	return r
}

func (r *Runner) WithRequestControlConfig(requestControlConfig *requestcontrol.Config) *Runner {
	r.requestControlConfig = requestControlConfig
	return r
}

func (r *Runner) WithSchedulerConfig(schedulerConfig *scheduling.SchedulerConfig) *Runner {
	r.schedulerConfig = schedulerConfig
	return r
}

func (r *Runner) WithCustomCollectors(collectors ...prometheus.Collector) *Runner {
	r.customCollectors = collectors
	return r
}

func (r *Runner) Run(ctx context.Context) error {
	// Setup a very basic logger in case command line argument parsing fails
	logutil.InitSetupLogging()

	setupLog.Info(r.eppExecutableName+" build", "commit-sha", version.CommitSHA, "build-ref", version.BuildRef)

	opts := runserver.NewOptions()
	opts.AddFlags(pflag.CommandLine)
	pflag.Parse()

	if err := opts.Complete(); err != nil {
		return err
	}
	if err := opts.Validate(); err != nil {
		setupLog.Error(err, "Failed to validate flags")
		return err
	}

	// Print flag values, skipping deprecated metric flags configured via engineConfigs
	flags := make(map[string]any)
	pflag.VisitAll(func(f *pflag.Flag) {
		if !runserver.IsDeprecatedMetricFlag(f.Name) {
			flags[f.Name] = f.Value
		}
	})
	setupLog.Info("Flags processed", "flags", flags)

	logutil.InitLogging(&opts.ZapOptions)

	if opts.Tracing {
		shutdown, err := tracing.InitTracing(ctx, setupLog, "llm-d-epp")
		if err != nil {
			return fmt.Errorf("failed to init tracing %w", err)
		}
		if shutdown != nil {
			defer func() {
				if err := shutdown(context.Background()); err != nil {
					setupLog.Error(err, "Failed to shutdown tracing")
				}
			}()
		}
	}

	// If the config specifies a discovery plugin, take the file discovery path which
	// does not require a Kubernetes cluster. Otherwise fall through to the K8s path.
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	if err != nil {
		setupLog.Error(err, "Failed to parse configuration")
		return err
	}
	if rawConfig.DataLayer != nil && rawConfig.DataLayer.Discovery != nil {
		return r.runWithFileDiscovery(ctx, opts, rawConfig)
	}

	// --- Get Kubernetes Config ---
	cfg, err := ctrl.GetConfig()
	if err != nil {
		setupLog.Error(err, "Failed to get Kubernetes rest config")
		return err
	}

	mgr, _, err := r.setup(ctx, cfg, opts, nil)
	if err != nil {
		return err
	}

	return r.runWithGracefulShutdown(ctx, mgr, opts.DrainTimeout)
}

// runWithGracefulShutdown runs the ext_proc and health servers on a context that
// outlives the manager. On SIGTERM (ctx cancelled) the manager stops and releases
// its leader lease, the pod is marked NotServing (so Kubernetes drains it from the
// Service endpoints), and the ext_proc server keeps accepting requests for
// drainTimeout so in-flight and pre-DNS-refresh requests are served rather than
// rejected. A drainTimeout of 0 stops the servers as soon as the manager
// terminates. setup() has stashed the servers on r.
func (r *Runner) runWithGracefulShutdown(ctx context.Context, mgr ctrl.Manager, drainTimeout time.Duration) error {
	// serveCtx is intentionally rooted at Background, not ctx, so SIGTERM does not
	// immediately stop the ext_proc/health servers.
	serveCtx, serveCancel := context.WithCancel(context.Background())
	defer serveCancel()

	serveErr := make(chan error, 1)
	go func() {
		g := newRunnableGroup()
		g.Add("ext-proc", func(c context.Context) error {
			return r.serverRunner.AsRunnable(ctrl.Log.WithName("ext-proc")).Start(c)
		})
		g.Add("health", func(c context.Context) error {
			return runnable.NoLeaderElection(runnable.GRPCServer("health", r.healthGRPCServer, r.healthGRPCPort)).Start(c)
		})
		serveErr <- g.Run(serveCtx)
	}()

	// Flip readiness to NotServing as soon as SIGTERM arrives (concurrently with
	// the manager's shutdown/lease release), so Kubernetes starts draining us from
	// the Service endpoints while we keep serving in-flight traffic.
	go func() {
		<-ctx.Done()
		r.draining.Store(true)
		setupLog.Info("Shutdown signal received: draining (NotServing) while finishing in-flight requests", "drainTimeout", drainTimeout)
	}()

	// Blocks until SIGTERM cancels ctx; returns once the manager has stopped and
	// released the leader lease (LeaderElectionReleaseOnCancel).
	setupLog.Info("Controller manager starting")
	mgrErr := mgr.Start(ctx)
	if mgrErr != nil {
		setupLog.Error(mgrErr, "Error starting controller manager")
		serveCancel()
		<-serveErr
		return mgrErr
	}
	setupLog.Info("Controller manager terminated; starting drain window")

	// Keep serving ext_proc for the drain window, then stop. GracefulStop drains
	// in-flight streams.
	select {
	case <-time.After(drainTimeout):
		setupLog.Info("Drain window elapsed, stopping ext_proc server")
	case err := <-serveErr:
		// The servers exited on their own (e.g. listener error) during the drain.
		return err
	}
	serveCancel()
	return <-serveErr
}

// setup configures the internal state of the Runner, including the manager,
// datastore, and other server components. It returns the initialized Manager
// without starting it, allowing for flexible use in integration tests.
//
// The returned Datastore is **only** meant to be used in the integration test.
// Optional managerOverrides are applied to the controller manager options before creation.

func (r *Runner) setup(ctx context.Context, cfg *rest.Config, opts *runserver.Options, managerOverrides []func(*ctrl.Options)) (ctrl.Manager, datastore.Datastore, error) {
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	if err != nil {
		setupLog.Error(err, "Failed to parse configuration")
		return nil, nil, err
	}
	setupLog.Info("Raw config after phase one", "config", toRawMap(rawConfig))

	epf := r.setupMetricsCollection(opts)
	gknn, err := extractGKNN(opts.PoolName, opts.PoolGroup, opts.PoolNamespace, opts.EndpointSelector)
	if err != nil {
		setupLog.Error(err, "Failed to extract GKNN")
		return nil, nil, err
	}

	startCrdReconcilers := opts.EndpointSelector == nil // If endpointSelector is nil, it means it's not in the standalone mode. Then we should start the inferencePool and other CRD Reconciler.
	controllerCfg := runserver.NewControllerConfig(startCrdReconcilers)
	controllerCfg.PopulateNonLeaderDatastore = r.featureGates[runserver.HAPopulateNonLeaderDatastoreFeatureGate]
	if err := controllerCfg.PopulateControllerConfig(cfg); err != nil {
		setupLog.Error(err, "Failed to populate controller config")
		return nil, nil, err
	}

	ds, err := setupDatastore(ctx, epf, startCrdReconcilers,
		gknn.Namespace, gknn.Name, opts.EndpointSelector, opts.EndpointTargetPorts)
	if err != nil {
		setupLog.Error(err, "Failed to setup datastore")
		return nil, nil, err
	}
	eppConfig, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds)
	if err != nil {
		setupLog.Error(err, "Failed to parse configuration")
		return nil, nil, err
	}
	setupLog.Info("EPP config after phase two", "config", eppConfig)

	// --- Setup Metrics Server ---
	r.customCollectors = append(r.customCollectors, collectors.NewInferencePoolMetricsCollector(ds))
	metrics.Register(r.customCollectors...)
	metrics.RecordInferenceExtensionInfo(version.CommitSHA, version.BuildRef)
	// Register metrics handler.
	// Metrics endpoint is enabled in 'config/default/kustomization.yaml'. The Metrics options configure the server.
	// More info:
	// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/metrics/server
	// - https://book.kubebuilder.io/reference/metrics.html
	metricsServerOptions := metricsserver.Options{
		BindAddress: fmt.Sprintf(":%d", opts.MetricsPort),
		FilterProvider: func() func(c *rest.Config, httpClient *http.Client) (metricsserver.Filter, error) {
			if opts.MetricsEndpointAuth {
				return filters.WithAuthenticationAndAuthorization
			}

			return nil
		}(),
	}

	if err := runserver.ConfigureMetricsTLS(opts, &metricsServerOptions); err != nil {
		return nil, nil, err
	}

	isLeader := &atomic.Bool{}
	isLeader.Store(false)

	mgr, err := runserver.NewDefaultManager(controllerCfg, *gknn, cfg, metricsServerOptions, opts.EnableLeaderElection, managerOverrides...)
	if err != nil {
		setupLog.Error(err, "Failed to create controller manager")
		return nil, nil, err
	}

	if opts.EnableLeaderElection {
		setupLog.Info("Leader election enabled")
		go func() {
			<-mgr.Elected()
			isLeader.Store(true)
			setupLog.Info("This instance is now the leader!")
		}()
	} else {
		// If leader election is disabled, all instances are "leaders" for readiness purposes.
		isLeader.Store(true)
	}

	if opts.EnablePprof {
		setupLog.Info("Setting pprof handlers")
		if err = profiling.SetupPprofHandlers(mgr); err != nil {
			setupLog.Error(err, "Failed to setup pprof handlers")
			return nil, nil, err
		}
	}

	if r.PluginHandle == nil {
		setupLog.Info("Plugin state debug handler not registered: plugin handle unavailable")
	} else if err = runserver.SetupPluginStateDebugHandler(mgr, r.PluginHandle); err != nil {
		setupLog.Error(err, "Failed to setup plugin state debug handler")
		return nil, nil, err
	}

	// --- Initialize Core EPP Components ---
	if r.schedulerConfig == nil {
		err := errors.New("scheduler config must be set either by config api or through code")
		setupLog.Error(err, "failed to create scheduler")
		return nil, nil, err
	}

	setupLog.Info("parsed config", "scheduler-config", r.schedulerConfig)

	scheduler := scheduling.NewSchedulerWithConfig(r.schedulerConfig)

	if err := r.configureAndStartDatalayer(ctx, eppConfig.DataConfig, mgr); err != nil {
		setupLog.Error(err, "failed to initialize data layer")
		return nil, nil, err
	}

	endpointCandidates := contracts.EndpointCandidates(requestcontrol.NewDatastoreEndpointCandidates(ds,
		requestcontrol.WithDisableEndpointSubsetFilter(opts.DisableEndpointSubsetFilter)))
	endpointCandidates, admissionController, priorityBandControlPlane := r.initAdmissionControl(ctx, opts, eppConfig, endpointCandidates)

	director := requestcontrol.NewDirectorWithConfig(ds, scheduler, admissionController, endpointCandidates, r.requestControlConfig)

	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:                         opts.GRPCPort,
		GKNN:                             *gknn,
		Datastore:                        ds,
		ControllerCfg:                    controllerCfg,
		SecureServing:                    opts.SecureServing,
		HealthChecking:                   opts.HealthChecking,
		CertPath:                         opts.CertPath,
		EnableCertReload:                 opts.EnableCertReload,
		RefreshPrometheusMetricsInterval: opts.RefreshPrometheusMetricsInterval,
		MetricsStalenessThreshold:        opts.MetricsStalenessThreshold,
		Director:                         director,
		ParserRegistry:                   r.parserRegistry,
		SaturationDetector:               eppConfig.SaturationDetector,
		PriorityBandControlPlane:         priorityBandControlPlane,
		GRPCMaxRecvMsgSize:               opts.GRPCMaxRecvMsgSize,
		GRPCMaxSendMsgSize:               opts.GRPCMaxSendMsgSize,
		EnableGRPCStreamMetrics:          opts.EnableGRPCStreamMetrics,
	}

	if err := serverRunner.SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "Failed to setup EPP controllers")
		return nil, nil, err
	}

	// --- Add Runnables to Manager ---
	// Register health server.
	parsers := r.parserRegistry.Parsers()
	supporters := make([]appProtocolSupporter, len(parsers))
	for i, p := range parsers {
		supporters[i] = p
	}

	// Keep the ext_proc and health servers OUT of the manager so they outlive
	// it. Run() runs them on a context that is only cancelled after the drain
	// window, so on SIGTERM the manager stops (releasing the leader lease) while
	// these keep serving. `draining` is flipped by Run() so readiness/ext_proc
	// health go NOT_SERVING and the pod is drained from the Service endpoints. A
	// DrainTimeout of 0 stops them as soon as the manager terminates.
	r.draining = &atomic.Bool{}
	r.serverRunner = serverRunner
	r.healthGRPCPort = opts.GRPCHealthPort
	r.healthGRPCServer = newHealthGRPCServer(ctrl.Log.WithName("health"), ds, isLeader, r.draining, opts.EnableLeaderElection, supporters)
	return mgr, ds, nil
}

// NewEndpointPoolFromOptions constructs an EndpointPool from standalone options.
// This is shared between the production runner and standalone integration tests.
func NewEndpointPoolFromOptions(
	namespace string,
	name string,
	endpointSelector labels.Selector,
	endpointTargetPorts []int,
) (*datalayer.EndpointPool, error) {
	if namespace == "" {
		return nil, errors.New("namespace must not be empty")
	}
	if name == "" {
		return nil, errors.New("name must not be empty")
	}
	if endpointSelector == nil {
		return nil, errors.New("endpoint selector must not be nil")
	}
	if len(endpointTargetPorts) == 0 {
		return nil, errors.New("endpoint target ports must not be empty")
	}

	pool := datalayer.NewEndpointPool(namespace, name)
	pool.Selector = endpointSelector
	pool.TargetPorts = append(pool.TargetPorts, endpointTargetPorts...)

	return pool, nil
}

func setupDatastore(ctx context.Context, epFactory datalayer.EndpointFactory,
	startCrdReconcilers bool, namespace, name string, endpointSelector labels.Selector, endpointTargetPorts []int) (datastore.Datastore, error) {

	if startCrdReconcilers {
		return datastore.NewDatastore(ctx, epFactory), nil
	}
	endpointPool, err := NewEndpointPoolFromOptions(namespace, name, endpointSelector, endpointTargetPorts)
	if err != nil {
		setupLog.Error(err, "Failed to construct endpoint pool from options")
		return nil, err
	}
	return datastore.NewDatastore(ctx, epFactory).WithEndpointPool(endpointPool), nil
}

// registerInTreePlugins registers the factory functions of all known plugins
func (r *Runner) registerInTreePlugins() {
	// bylabel role filters
	// Beta
	fwkplugin.Register(bylabel.LabelSelectorFilterType, fwkplugin.StabilityBeta, bylabel.SelectorFactory)
	fwkplugin.RegisterDeprecated(bylabel.ByLabelSelectorType, fwkplugin.StabilityBeta, bylabel.DeprecatedSelectorFactory, "v0.9.0", "v0.11.0", bylabel.LabelSelectorFilterType) //nolint:staticcheck // deprecated in v0.9.0 (#842)
	fwkplugin.RegisterDeprecated(bylabel.ByLabelType, fwkplugin.StabilityBeta, bylabel.Factory, "v0.9.0", "v0.11.0", bylabel.LabelSelectorFilterType)                           //nolint:staticcheck // deprecated in v0.9.0 (#842)
	fwkplugin.Register(bylabel.EncodeRoleType, fwkplugin.StabilityBeta, bylabel.EncodeRoleFactory)
	fwkplugin.Register(bylabel.DecodeRoleType, fwkplugin.StabilityBeta, bylabel.DecodeRoleFactory)
	fwkplugin.Register(bylabel.PrefillRoleType, fwkplugin.StabilityBeta, bylabel.PrefillRoleFactory)
	// Alpha
	fwkplugin.Register(sessionaffinityfilter.SessionAffinityType, fwkplugin.StabilityAlpha, sessionaffinityfilter.Factory)
	fwkplugin.Register(endpointattributefilter.EndpointAttributeFilterType, fwkplugin.StabilityAlpha, endpointattributefilter.EndpointAttributeFilterFactory)

	// dataparallel profile handler
	// Beta
	fwkplugin.Register(dataparallel.DataParallelProfileHandlerType, fwkplugin.StabilityBeta, dataparallel.ProfileHandlerFactory)

	// extra scheduling scorers
	// Beta
	fwkplugin.Register(loadaware.LoadAwareType, fwkplugin.StabilityBeta, loadaware.Factory)
	fwkplugin.Register(contextlengthaware.ContextLengthAwareType, fwkplugin.StabilityBeta, contextlengthaware.Factory)
	// Alpha
	fwkplugin.Register(sessionaffinity.SessionAffinityType, fwkplugin.StabilityAlpha, sessionaffinity.Factory)

	// data layer models source/extractor
	// Beta
	fwkplugin.Register(srcmodels.ModelsDataSourceType, fwkplugin.StabilityBeta, srcmodels.ModelDataSourceFactory)
	fwkplugin.Register(attrmodels.ModelsExtractorType, fwkplugin.StabilityBeta, extmodels.ModelServerExtractorFactory)
	// Alpha
	fwkplugin.Register(attrtopology.TopologyExtractorType, fwkplugin.StabilityAlpha, exttopology.Factory)

	// data layer DCGM source/extractor
	// Alpha
	fwkplugin.Register(srcdcgm.DCGMDataSourceType, fwkplugin.StabilityAlpha, srcdcgm.DCGMDataSourceFactory)
	fwkplugin.Register(attrgpu.DCGMExtractorType, fwkplugin.StabilityAlpha, extdcgm.DCGMExtractorFactory)

	// scheduling & profile handler plugins
	// Beta
	fwkplugin.Register(prefix.PrefixCacheScorerPluginType, fwkplugin.StabilityBeta, prefix.PrefixCachePluginFactory)
	fwkplugin.Register(maxscore.MaxScorePickerType, fwkplugin.StabilityBeta, maxscore.MaxScorePickerFactory)
	fwkplugin.Register(random.RandomPickerType, fwkplugin.StabilityBeta, random.RandomPickerFactory)
	fwkplugin.Register(weightedrandom.WeightedRandomPickerType, fwkplugin.StabilityBeta, weightedrandom.WeightedRandomPickerFactory)
	fwkplugin.Register(single.SingleProfileHandlerType, fwkplugin.StabilityBeta, single.SingleProfileHandlerFactory)
	fwkplugin.RegisterDeprecated(disagg.DisaggHeadersHandlerType, fwkplugin.StabilityBeta, disagg.HeadersHandlerFactory, "v0.9.0", "v0.11.0", disagg.DisaggProfileHandlerType) //nolint:staticcheck // deprecated in v0.9.0 (#905)
	fwkplugin.RegisterDeprecated(disagg.PrefillHeaderHandlerType, fwkplugin.StabilityBeta, disagg.HeadersHandlerFactory, "v0.9.0", "v0.11.0", disagg.DisaggProfileHandlerType) //nolint:staticcheck // deprecated in v0.9.0 (#905)
	fwkplugin.RegisterDeprecatedWithPluginDependencies(disagg.PdProfileHandlerType, fwkplugin.StabilityBeta, disagg.PdProfileHandlerFactory,                                   //nolint:staticcheck // deprecated in v0.7.0 (#732/#756)
		disagg.PdProfileHandlerConfigParser, "v0.7.0", "v0.9.0", disagg.DisaggProfileHandlerType)
	fwkplugin.RegisterWithPluginDependencies(disagg.DisaggProfileHandlerType, fwkplugin.StabilityBeta, disagg.HandlerFactory, disagg.DisaggProfileHandlerConfigParser)
	fwkplugin.Register(disagg.AlwaysDisaggPDDeciderPluginType, fwkplugin.StabilityBeta, disagg.AlwaysDisaggPDDeciderPluginFactory)
	fwkplugin.Register(disagg.PrefixBasedPDDeciderPluginType, fwkplugin.StabilityBeta, disagg.PrefixBasedPDDeciderPluginFactory)
	fwkplugin.Register(disagg.AlwaysDisaggMulimodalPluginType, fwkplugin.StabilityBeta, disagg.AlwaysDisaggMulimodalDeciderPluginFactory)
	fwkplugin.Register(kvcacheutilization.KvCacheUtilizationScorerType, fwkplugin.StabilityBeta, kvcacheutilization.KvCacheUtilizationScorerFactory)
	fwkplugin.Register(queuedepth.QueueScorerType, fwkplugin.StabilityBeta, queuedepth.QueueScorerFactory)
	fwkplugin.Register(runningrequests.RunningRequestsSizeScorerType, fwkplugin.StabilityBeta, runningrequests.RunningRequestsSizeScorerFactory)
	fwkplugin.Register(loraaffinity.LoraAffinityScorerType, fwkplugin.StabilityBeta, loraaffinity.LoraAffinityScorerFactory)
	fwkplugin.Register(tokenload.TokenLoadScorerType, fwkplugin.StabilityBeta, tokenload.TokenLoadScorerFactory)
	fwkplugin.Register(nohitlru.NoHitLRUType, fwkplugin.StabilityBeta, nohitlru.Factory)
	fwkplugin.Register(activerequest.ActiveRequestType, fwkplugin.StabilityBeta, activerequest.Factory)
	fwkplugin.Register(preciseprefixcache.PrecisePrefixCachePluginType, fwkplugin.StabilityBeta, preciseprefixcache.PluginFactory)
	fwkplugin.Register(mmcacheaffinity.Type, fwkplugin.StabilityBeta, mmcacheaffinity.Factory)
	fwkplugin.Register(preciseproducer.PluginType, fwkplugin.StabilityBeta, preciseproducer.PluginFactory)
	// Alpha
	fwkplugin.Register(endpointattribute.EndpointAttributeScorerType, fwkplugin.StabilityAlpha, endpointattribute.EndpointAttributeScorerFactory)
	fwkplugin.Register(burstprefix.PluginType, fwkplugin.StabilityAlpha, burstprefix.Factory)
	fwkplugin.Register(p2psource.PluginType, fwkplugin.StabilityAlpha, p2psource.PluginFactory)

	// Flow Control plugins
	// Beta
	fwkplugin.Register(globalstrict.GlobalStrictFairnessPolicyType, fwkplugin.StabilityBeta, globalstrict.GlobalStrictFairnessPolicyFactory)
	fwkplugin.Register(roundrobin.RoundRobinFairnessPolicyType, fwkplugin.StabilityBeta, roundrobin.RoundRobinFairnessPolicyFactory)
	fwkplugin.Register(programaware.ProgramAwarePluginType, fwkplugin.StabilityBeta, programaware.ProgramAwarePluginFactory)
	fwkplugin.Register(fcfs.FCFSOrderingPolicyType, fwkplugin.StabilityBeta, fcfs.FCFSOrderingPolicyFactory)
	fwkplugin.Register(edf.EDFOrderingPolicyType, fwkplugin.StabilityBeta, edf.EDFOrderingPolicyFactory)
	fwkplugin.Register(slodeadline.SLODeadlineOrderingPolicyType, fwkplugin.StabilityBeta, slodeadline.SLODeadlineOrderingPolicyFactory)
	fwkplugin.Register(usagelimits.StaticUsageLimitPolicyType, fwkplugin.StabilityBeta, usagelimits.StaticPolicyFactory)
	// Alpha
	fwkplugin.Register(evictfiltering.SheddableFilterType, fwkplugin.StabilityAlpha, evictfiltering.SheddableFilterFactory)
	fwkplugin.Register(evictordering.PriorityThenTimeOrderingType, fwkplugin.StabilityAlpha, evictordering.PriorityThenTimeOrderingFactory)
	fwkplugin.Register(priorityholdback.PolicyType, fwkplugin.StabilityAlpha, priorityholdback.PolicyFactory)
	fwkplugin.Register(softreflectiveceiling.PolicyType, fwkplugin.StabilityAlpha, softreflectiveceiling.Factory)

	// Register Request level data producer plugins as defaults for their respective data keys.
	// Beta
	fwkplugin.RegisterAsDefaultProducer(reqdataprodprefix.ApproxPrefixCachePluginType, fwkplugin.StabilityBeta, reqdataprodprefix.ApproxPrefixCacheFactory, attrprefix.PrefixCacheMatchInfoDataKey)
	fwkplugin.RegisterAsDefaultProducer(inflightload.InFlightLoadProducerType, fwkplugin.StabilityBeta, inflightload.InFlightLoadProducerFactory, attrconcurrency.InFlightLoadDataKey)
	fwkplugin.RegisterAsDefaultProducer(latencyproducer.LatencyDataProviderPluginType, fwkplugin.StabilityBeta, latencyproducer.PredictedLatencyFactory, attrlatency.LatencyPredictionInfoDataKey)
	fwkplugin.RegisterDeprecated(tokenizer.LegacyPluginType, fwkplugin.StabilityBeta, tokenizer.LegacyPluginFactory, "v0.9.0", "v0.11.0", tokenizer.PluginType) //nolint:staticcheck // deprecated in v0.9.0 (#863)
	fwkplugin.RegisterAsDefaultProducer(mmproducer.ProducerType, fwkplugin.StabilityBeta, mmproducer.Factory, mmproducer.ProducedKey)
	fwkplugin.RegisterAsDefaultProducer(tokenizer.PluginType, fwkplugin.StabilityBeta, tokenizer.PluginFactory, tokenizer.TokenizedPromptDataKey)
	fwkplugin.RegisterAsDefaultProducer(sessionid.SessionIDProducerType, fwkplugin.StabilityBeta, sessionid.Factory, attrsession.SessionIDDataKey)

	// Latency predictor plugins
	// Beta
	fwkplugin.Register(latencyslo.LatencyAdmissionPluginType, fwkplugin.StabilityBeta, latencyslo.LatencyAdmissionFactory)
	fwkplugin.Register(probabilisticadmitter.Type, fwkplugin.StabilityBeta, probabilisticadmitter.Factory)

	// Latency scoring and filtering plugins
	// Beta
	fwkplugin.Register(prefixcacheaffinity.PluginType, fwkplugin.StabilityBeta, prefixcacheaffinity.Factory)
	fwkplugin.Register(sloheadroomtier.PluginType, fwkplugin.StabilityBeta, sloheadroomtier.Factory)
	fwkplugin.Register(latencyscorer.LatencyScorerType, fwkplugin.StabilityBeta, latencyscorer.Factory)
	fwkplugin.Register(bylabel.PrefillRoleType, fwkplugin.StabilityBeta, bylabel.PrefillRoleFactory)
	fwkplugin.Register(bylabel.DecodeRoleType, fwkplugin.StabilityBeta, bylabel.DecodeRoleFactory)

	// register filter for test purpose only (used in conformance tests)
	// Beta
	fwkplugin.Register(testfilter.HeaderBasedTestingFilterType, fwkplugin.StabilityBeta, testfilter.HeaderBasedTestingFilterFactory)
	fwkplugin.Register(testresponsereceived.DestinationEndpointServedVerifierType, fwkplugin.StabilityBeta, testresponsereceived.DestinationEndpointServedVerifierFactory)

	// register datalayer metrics collection plugins
	// Beta
	fwkplugin.Register(sourcemetrics.MetricsDataSourceType, fwkplugin.StabilityBeta, sourcemetrics.MetricsDataSourceFactory)
	fwkplugin.Register(extractormetrics.MetricsExtractorType, fwkplugin.StabilityBeta, extractormetrics.CoreMetricsExtractorFactory)

	// register datalayer notification source plugins
	// Beta
	fwkplugin.Register(sourcenotifications.NotificationSourceType, fwkplugin.StabilityBeta, sourcenotifications.NotificationSourceFactory)
	fwkplugin.Register(sourcenotifications.EndpointNotificationSourceType, fwkplugin.StabilityBeta, sourcenotifications.EndpointSourceFactory)

	// register request control plugins
	// Beta
	fwkplugin.Register(requestattributereporter.RequestAttributeReporterType, fwkplugin.StabilityBeta, requestattributereporter.RequestAttributeReporterPluginFactory)
	fwkplugin.Register(openai.OpenAIParserType, fwkplugin.StabilityBeta, openai.OpenAIParserPluginFactory)
	fwkplugin.Register(vllmgrpc.VllmGRPCParserType, fwkplugin.StabilityBeta, vllmgrpc.VllmGRPCParserPluginFactory)
	fwkplugin.Register(passthrough.PassthroughParserType, fwkplugin.StabilityBeta, passthrough.PassthroughParserPluginFactory)
	fwkplugin.Register(anthropic.AnthropicParserType, fwkplugin.StabilityBeta, anthropic.AnthropicParserPluginFactory)
	fwkplugin.Register(vllmhttp.VllmHTTPParserType, fwkplugin.StabilityBeta, vllmhttp.VllmHTTPParserPluginFactory)
	fwkplugin.Register(vertexai.VertexAIParserType, fwkplugin.StabilityBeta, vertexai.VertexAIParserPluginFactory)

	// register saturation detector plugins
	// Beta
	fwkplugin.Register(concurrency.ConcurrencyDetectorType, fwkplugin.StabilityBeta, concurrency.ConcurrencyDetectorFactory)
	fwkplugin.Register(utilization.UtilizationDetectorType, fwkplugin.StabilityBeta, utilization.UtilizationDetectorFactory)

	// register discovery plugins
	// Beta
	fwkplugin.Register(discoveryfile.PluginType, fwkplugin.StabilityBeta, discoveryfile.Factory)

	// register request header processor plugins
	// Alpha
	fwkplugin.Register(agentidentity.PluginType, fwkplugin.StabilityAlpha, agentidentity.PluginFactory)
}

func (r *Runner) parseConfigurationPhaseOne(ctx context.Context, opts *runserver.Options) (*configapi.EndpointPickerConfig, error) {
	// parseConfigurationPhaseOne is idempotent: Run() calls it to decide
	// between the K8s and file-discovery paths, and the K8s path's setup()
	// then calls it a second time. Cache the parsed config so we don't
	// re-read the file, re-register plugins/feature gates, or re-emit the
	// data-layer setup logs.
	if r.rawConfig != nil {
		return r.rawConfig, nil
	}

	logger := log.FromContext(ctx)

	var configBytes []byte
	if opts.ConfigText != "" {
		configBytes = []byte(opts.ConfigText)
	} else if opts.ConfigFile != "" { // if config was specified through a file
		var err error
		configBytes, err = os.ReadFile(opts.ConfigFile)
		if err != nil {
			return nil, fmt.Errorf("failed to load config from a file '%s' - %w", opts.ConfigFile, err)
		}
	}

	loader.RegisterFeatureGate(flowcontrol.FeatureGate, false)
	loader.RegisterFeatureGate(runserver.HAPopulateNonLeaderDatastoreFeatureGate, true)
	loader.RegisterFeatureGate(fwkplugin.ExperimentalPluginsFeatureGate, false)

	r.registerInTreePlugins()

	rawConfig, featureGates, err := loader.LoadRawConfig(configBytes, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to parse config - %w", err)
	}

	r.featureGates = featureGates

	setupLog.Info("Data layer: ENABLED")

	r.rawConfig = rawConfig
	return rawConfig, nil
}

// Return a function that can be used in the EPP Handle to list pod names.
func makePodListFunc(ds datastore.Datastore) func() []types.NamespacedName {
	return func() []types.NamespacedName {
		pods := ds.PodList(datastore.AllPodsPredicate)
		names := make([]types.NamespacedName, 0, len(pods))

		for _, p := range pods {
			names = append(names, p.GetMetadata().NamespacedName)
		}
		return names
	}
}

func (r *Runner) parseConfigurationPhaseTwo(ctx context.Context, rawConfig *configapi.EndpointPickerConfig, ds datastore.Datastore) (*config.Config, error) {
	logger := log.FromContext(ctx)

	applyDeprecatedEnvFeatureGate(enableExperimentalFlowControlLayer, "Flow Control layer", flowcontrol.FeatureGate, rawConfig)

	handle := fwkplugin.NewEppHandle(ctx, makePodListFunc(ds), fwkplugin.WithMetricsRecorder(ctrlmetrics.Registry))
	r.PluginHandle = handle
	cfg, err := loader.InstantiateAndConfigure(rawConfig, handle, logger)

	if err != nil {
		return nil, fmt.Errorf("failed to load the configuration - %w", err)
	}

	r.schedulerConfig = cfg.SchedulerConfig

	// Auto-create any DataProducer plugins that are needed by consumers already in
	// the config but not yet satisfied by an existing producer.
	if err := datalayer.CreateMissingDataProducers(ctx, fwkplugin.DefaultProducerRegistry, fwkplugin.Registry, handle); err != nil {
		return nil, fmt.Errorf("failed to create missing data producers - %w", err)
	}

	// Add requestControl plugins
	r.requestControlConfig.AddPlugins(handle.GetAllPlugins()...)

	// Let plugins declare their datalayer source/extractor dependencies before Configure().
	for _, p := range handle.GetAllPlugins() {
		if registrant, ok := p.(fwkdl.Registrant); ok {
			if err := registrant.RegisterDependencies(r.dlRuntime); err != nil {
				return nil, fmt.Errorf("plugin %s RegisterDependencies: %w", p.TypedName(), err)
			}
		}
	}

	// Sort data plugins in DAG order (topological sort). Also check DAG for cycles.
	// This must run after auto-created producers are added so they are included in the ordering.
	dag, err := datalayer.ValidateAndOrderDataDependencies(handle.GetAllPlugins())
	if err != nil {
		return nil, fmt.Errorf("failed to load the configuration - %w", err)
	}

	// The plugins will be executed in topologically sorted order to ensure that data is produced before it is consumed.
	r.requestControlConfig.OrderDataProducerPlugins(dag)

	r.parserRegistry = cfg.ParserRegistry
	logger.Info("loaded configuration from file/text successfully")

	return cfg, nil
}

func applyDeprecatedEnvFeatureGate(envVar, featureName, featureGate string, rawConfig *configapi.EndpointPickerConfig) {
	if _, ok := os.LookupEnv(envVar); ok {
		setupLog.Info(fmt.Sprintf("Enabling the experimental %s using environment variables is deprecated and will be removed in next version", featureName))
		if env.GetEnvBool(envVar, false, setupLog) {
			if rawConfig.FeatureGates == nil {
				rawConfig.FeatureGates = make(configapi.FeatureGates, 0)
			}
			rawConfig.FeatureGates = append(rawConfig.FeatureGates, featureGate)
		}
	}
}

func (r *Runner) configureAndStartDatalayer(ctx context.Context, cfg *datalayer.Config, mgr ctrl.Manager) error {
	if err := r.dlRuntime.Configure(cfg, setupLog); err != nil {
		return err
	}

	return r.dlRuntime.Start(ctx, mgr)
}

func (r *Runner) setupMetricsCollection(opts *runserver.Options) datalayer.EndpointFactory {
	r.dlRuntime = datalayer.NewRuntime(opts.RefreshMetricsInterval)
	return r.dlRuntime
}

// newHealthGRPCServer builds the gRPC health server. draining may be nil (graceful
// drain disabled); when non-nil and set, non-liveness checks report NOT_SERVING.
func newHealthGRPCServer(logger logr.Logger, ds datastore.Datastore, isLeader, draining *atomic.Bool, leaderElectionEnabled bool, supporters []appProtocolSupporter) *grpc.Server {
	srv := grpc.NewServer()
	healthPb.RegisterHealthServer(srv, &healthServer{
		logger:                logger,
		datastore:             ds,
		isLeader:              isLeader,
		leaderElectionEnabled: leaderElectionEnabled,
		supporters:            supporters,
		draining:              draining,
	})
	return srv
}

func extractDeploymentName(podName string) (string, error) {
	regex := regexp.MustCompile(`^(.+)-[a-z0-9]+-[a-z0-9]+$`)

	matches := regex.FindStringSubmatch(podName)
	if len(matches) == 2 {
		return matches[1], nil
	}
	return "", fmt.Errorf("failed to parse deployment name from pod name %s", podName)
}

func extractGKNN(poolName, poolGroup, poolNamespace string, endpointSelector labels.Selector) (*common.GKNN, error) {
	if poolName != "" {
		resolvedPoolNamespace := resolvePoolNamespace(poolNamespace)
		poolNamespacedName := types.NamespacedName{
			Name:      poolName,
			Namespace: resolvedPoolNamespace,
		}
		poolGroupKind := schema.GroupKind{
			Group: poolGroup,
			Kind:  "InferencePool",
		}
		return &common.GKNN{
			NamespacedName: poolNamespacedName,
			GroupKind:      poolGroupKind,
		}, nil
	}

	if endpointSelector != nil {
		resolvedPoolNamespace := resolvePoolNamespace(poolNamespace)
		eppPodNameEnv := os.Getenv("POD_NAME")
		if eppPodNameEnv == "" {
			return nil, errors.New("failed to get environment variable POD_NAME")

		}
		eppName, err := extractDeploymentName(eppPodNameEnv)
		if err != nil {
			return nil, err
		}
		return &common.GKNN{
			NamespacedName: types.NamespacedName{Namespace: resolvedPoolNamespace, Name: eppName},
			GroupKind:      schema.GroupKind{Kind: "Deployment", Group: "apps"},
		}, nil
	}
	return nil, errors.New("can't construct gknn as both pool-name and endpoint-selector are missing")
}

func resolvePoolNamespace(poolNamespace string) string {
	if poolNamespace != "" {
		return poolNamespace
	}
	if nsEnv := os.Getenv("NAMESPACE"); nsEnv != "" {
		return nsEnv
	}
	return runserver.DefaultPoolNamespace
}

// resolveDiscovery returns the discovery plugin identified by
// rawConfig.DataLayer.Discovery.PluginRef. The plugin is expected to have
// already been instantiated and registered in r.PluginHandle by
// parseConfigurationPhaseTwo; this function only looks it up and verifies its
// type, so the loader-created instance (with its real Handle wired in) is the
// one the runner drives.
func (r *Runner) resolveDiscovery(rawConfig *configapi.EndpointPickerConfig) (fwkdl.EndpointDiscovery, error) {
	ref := rawConfig.DataLayer.Discovery.PluginRef
	p := r.PluginHandle.Plugin(ref)
	if p == nil {
		return nil, fmt.Errorf("discovery: no plugin found with name %q", ref)
	}
	disc, ok := p.(fwkdl.EndpointDiscovery)
	if !ok {
		return nil, fmt.Errorf("discovery: plugin %q does not implement EndpointDiscovery", ref)
	}
	return disc, nil
}

// initAdmissionControl builds the request admission controller, gated by the
// FlowControl feature gate. With FC on it constructs the FlowRegistry and
// FlowController and wraps endpointCandidates in a short-lived cache; with FC
// off it returns the legacy saturation-only controller. Shared by the K8s and
// file-discovery startup paths so the two cannot drift.
func (r *Runner) initAdmissionControl(
	ctx context.Context,
	opts *runserver.Options,
	eppConfig *config.Config,
	endpointCandidates contracts.EndpointCandidates,
) (contracts.EndpointCandidates, requestcontrol.AdmissionController, contracts.PriorityBandControlPlane) {
	if !r.featureGates[flowcontrol.FeatureGate] {
		setupLog.Info("Experimental Flow Control layer is disabled, using legacy admission control")
		return endpointCandidates,
			requestcontrol.NewLegacyAdmissionController(eppConfig.SaturationDetector, endpointCandidates),
			nil
	}
	endpointCandidates = requestcontrol.NewCachedEndpointCandidates(ctx, endpointCandidates, 50*time.Millisecond)
	setupLog.Info("Initializing experimental Flow Control layer")
	registry := fcregistry.NewFlowRegistry(eppConfig.FlowControlConfig.Registry, setupLog)
	fc := fccontroller.NewFlowController(
		ctx,
		opts.PoolName,
		eppConfig.FlowControlConfig.Controller,
		fccontroller.Deps{
			Registry:           registry,
			SaturationDetector: eppConfig.SaturationDetector,
			EndpointCandidates: endpointCandidates,
			UsageLimitPolicy:   eppConfig.FlowControlConfig.UsageLimitPolicy,
		},
	)
	return endpointCandidates, requestcontrol.NewFlowControlAdmissionController(fc, opts.PoolName), registry
}

// runWithFileDiscovery handles the execution path when a discovery plugin is configured.
// It builds the EPP server stack without a Kubernetes cluster or controller manager.
func (r *Runner) runWithFileDiscovery(ctx context.Context, opts *runserver.Options, rawConfig *configapi.EndpointPickerConfig) error {
	epf := r.setupMetricsCollection(opts)

	namespace := resolvePoolNamespace(opts.PoolNamespace)
	poolName := opts.PoolName
	if poolName == "" {
		poolName = "epp"
	}
	pool := datalayer.NewEndpointPool(namespace, poolName)
	ds := datastore.NewDatastore(ctx, epf).WithEndpointPool(pool)

	// On bare metal / Slurm / Ray (or any deployment without the K8s Downward
	// API), neither --pool-namespace nor the NAMESPACE env var is set, so the
	// pool ends up labeled with the literal "default" -- a Kubernetes-flavored
	// string that is meaningless outside K8s. Behavior is unaffected; only
	// metrics labels and log fields look wrong. Warn so operators set
	// --pool-namespace explicitly for their environment.
	if opts.PoolNamespace == "" && os.Getenv("NAMESPACE") == "" {
		setupLog.Info("file-discovery mode: pool namespace defaulted to "+
			runserver.DefaultPoolNamespace+"; pass --pool-namespace to label "+
			"metrics and logs for your environment",
			"namespace", runserver.DefaultPoolNamespace)
	}

	// File mode runs without a controller manager, so several Kubernetes-only
	// features are inactive: the InferenceModelRewrite and InferenceObjective
	// reconcilers never start, and any "k8s-notification-source" plugin in the
	// data layer config silently fails to bind (Runtime.Start, which wires
	// notification sources into the manager, is intentionally skipped below).
	// Surface this once at startup so operators porting a K8s config see why
	// related behavior differs.
	//
	// Note on InferenceObjective: with no objective CRDs to consult, per-request
	// priority falls back to Director.defaultPriority (see
	// pkg/epp/requestcontrol/director.go). Static priority bands configured in
	// EndpointPickerConfig.flowControl are honored and applied via the FlowControl
	// layer when the feature gate is enabled.
	setupLog.Info("file-discovery mode: Kubernetes-only features are inactive " +
		"(InferenceModelRewrite, InferenceObjective reconciler, and any " +
		"k8s-notification-source data layer plugins); see docs/discovery.md")

	eppConfig, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds)
	if err != nil {
		setupLog.Error(err, "Failed to parse configuration")
		return err
	}

	disc, err := r.resolveDiscovery(rawConfig)
	if err != nil {
		setupLog.Error(err, "Failed to resolve discovery plugin")
		return err
	}

	if err := r.dlRuntime.Configure(eppConfig.DataConfig, setupLog); err != nil {
		return fmt.Errorf("failed to configure datalayer: %w", err)
	}

	if r.schedulerConfig == nil {
		return errors.New("scheduler config must be set either by config api or through code")
	}
	setupLog.Info("parsed config", "scheduler-config", r.schedulerConfig)

	scheduler := scheduling.NewSchedulerWithConfig(r.schedulerConfig)

	// Outside Kubernetes there is no InferenceObjective CRD, so per-request
	// priority falls back to Director.defaultPriority (see
	// pkg/epp/requestcontrol/director.go); static bands defined in
	// EndpointPickerConfig.flowControl still apply.
	endpointCandidates := contracts.EndpointCandidates(requestcontrol.NewDatastoreEndpointCandidates(ds,
		requestcontrol.WithDisableEndpointSubsetFilter(opts.DisableEndpointSubsetFilter)))
	// File-discovery mode has no InferenceObjective reconciler to drive the
	// control plane; static bands from config apply at registry construction.
	endpointCandidates, admissionController, _ := r.initAdmissionControl(ctx, opts, eppConfig, endpointCandidates)
	director := requestcontrol.NewDirectorWithConfig(ds, scheduler, admissionController, endpointCandidates, r.requestControlConfig)

	gknn := common.GKNN{
		NamespacedName: types.NamespacedName{Name: poolName, Namespace: namespace},
	}
	serverRunner := &runserver.ExtProcServerRunner{
		GrpcPort:                         opts.GRPCPort,
		GKNN:                             gknn,
		Datastore:                        ds,
		ControllerCfg:                    runserver.NewControllerConfig(false),
		SecureServing:                    opts.SecureServing,
		HealthChecking:                   opts.HealthChecking,
		CertPath:                         opts.CertPath,
		EnableCertReload:                 opts.EnableCertReload,
		RefreshPrometheusMetricsInterval: opts.RefreshPrometheusMetricsInterval,
		MetricsStalenessThreshold:        opts.MetricsStalenessThreshold,
		Director:                         director,
		ParserRegistry:                   r.parserRegistry,
		SaturationDetector:               eppConfig.SaturationDetector,
		GRPCMaxRecvMsgSize:               opts.GRPCMaxRecvMsgSize,
		GRPCMaxSendMsgSize:               opts.GRPCMaxSendMsgSize,
		EnableGRPCStreamMetrics:          opts.EnableGRPCStreamMetrics,
	}

	r.customCollectors = append(r.customCollectors, collectors.NewInferencePoolMetricsCollector(ds))
	metrics.Register(r.customCollectors...)
	metrics.RecordInferenceExtensionInfo(version.CommitSHA, version.BuildRef)

	setupLog.Info("EPP starting (file discovery mode)",
		"grpcPort", opts.GRPCPort,
		"pool", poolName,
		"namespace", namespace,
		"discoveryPlugin", disc.TypedName())

	isLeader := &atomic.Bool{}
	isLeader.Store(true)

	healthSrv := grpc.NewServer()
	parsers := r.parserRegistry.Parsers()
	ps := make([]appProtocolSupporter, len(parsers))
	for i, p := range parsers {
		ps[i] = p
	}
	healthPb.RegisterHealthServer(healthSrv, &healthServer{
		logger:                ctrl.Log.WithName("health"),
		datastore:             ds,
		isLeader:              isLeader,
		leaderElectionEnabled: false,
		supporters:            ps,
	})

	g := newRunnableGroup()
	g.Add("discovery", func(ctx context.Context) error {
		return disc.Start(ctx, fwkdl.NewDiscoveryNotifier(ds))
	})
	// epp-server and health wait for the discovery plugin's initial sync before
	// going live, so requests and probes never observe an empty datastore. See
	// EndpointDiscovery.Ready contract.
	g.Add("epp-server", func(ctx context.Context) error {
		select {
		case <-disc.Ready():
		case <-ctx.Done():
			return ctx.Err()
		}
		return serverRunner.AsRunnable(ctrl.Log.WithName("ext-proc")).Start(ctx)
	})
	g.Add("health", func(ctx context.Context) error {
		select {
		case <-disc.Ready():
		case <-ctx.Done():
			return ctx.Err()
		}
		return runnable.NoLeaderElection(runnable.GRPCServer("health", healthSrv, opts.GRPCHealthPort)).Start(ctx)
	})
	g.Add("metrics", func(ctx context.Context) error {
		return serveMetrics(ctx, opts.MetricsPort, opts.EnablePprof)
	})
	return g.Run(ctx)
}

// metricsShutdownTimeout bounds graceful shutdown of the metrics server so a
// scraper holding a connection at process exit cannot block termination.
const metricsShutdownTimeout = 5 * time.Second

// serveMetrics starts a standalone Prometheus metrics HTTP server.
//
// EPP metrics are registered with controller-runtime's registry (see
// pkg/epp/metrics.Register), not the prometheus default registry. The handler
// must serve ctrlmetrics.Registry directly; promhttp.Handler() would expose only
// Go runtime/process metrics and silently omit every EPP metric.
func serveMetrics(ctx context.Context, port int, enablePprof bool) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(ctrlmetrics.Registry, promhttp.HandlerOpts{EnableOpenMetrics: true}))
	if enablePprof {
		for path, h := range profiling.PprofHandlers() {
			mux.Handle(path, h)
		}
	}
	srv := &http.Server{Addr: fmt.Sprintf(":%d", port), Handler: mux}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), metricsShutdownTimeout)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("metrics server: %w", err)
	}
	return nil
}

func toRawMap(cfg *configapi.EndpointPickerConfig) map[string]any {
	if cfg == nil {
		return nil
	}
	var rawMap map[string]any
	bytes, err := json.Marshal(cfg)
	if err != nil {
		return map[string]any{
			"error": fmt.Sprintf("failed to marshal raw config: %v", err),
		}
	}
	if err := json.Unmarshal(bytes, &rawMap); err != nil {
		return map[string]any{
			"error": fmt.Sprintf("failed to unmarshal raw config map: %v", err),
		}
	}
	return rawMap
}
