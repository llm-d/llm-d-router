// Package plugins registers all llm-d inference scheduler plugins.
package plugins

import (
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	attrconcurrency "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	extmodels "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/extractor/models"
	srcmodels "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/source/models"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	mmproducer "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/requestcontrol/dataproducer/multimodal"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/profilehandler/dataparallel"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/activerequest"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/contextlengthaware"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/loadaware"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/mmcacheaffinity"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/nohitlru"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/preciseprefixcache"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/sessionaffinity"
)

// RegisterAllPlugins registers the factory functions of all plugins in this repository.
func RegisterAllPlugins() {
	plugin.Register(bylabel.LabelSelectorFilterType, bylabel.SelectorFactory)
	// Legacy aliases - existing YAML configs using by-label-selector or by-label continue to work.
	plugin.Register(bylabel.ByLabelSelectorType, bylabel.DeprecatedSelectorFactory) //nolint:staticcheck // intentional: keep backward compatibility
	plugin.Register(bylabel.ByLabelType, bylabel.Factory)                           //nolint:staticcheck // intentional: keep backward compatibility
	plugin.Register(bylabel.EncodeRoleType, bylabel.EncodeRoleFactory)
	plugin.Register(bylabel.DecodeRoleType, bylabel.DecodeRoleFactory)
	plugin.Register(bylabel.PrefillRoleType, bylabel.PrefillRoleFactory)
	// Deprecated aliases — disagg-profile-handler now handles PreRequest natively.
	plugin.Register(disagg.DisaggHeadersHandlerType, disagg.HeadersHandlerFactory) //nolint:staticcheck // intentional: keep backward compatibility
	plugin.Register(disagg.PrefillHeaderHandlerType, disagg.HeadersHandlerFactory) //nolint:staticcheck // intentional: keep backward compatibility
	plugin.Register(dataparallel.DataParallelProfileHandlerType, dataparallel.ProfileHandlerFactory)
	plugin.Register(disagg.DisaggProfileHandlerType, disagg.HandlerFactory)
	// Legacy aliases - existing YAML configs continue to work.
	// golangci-lint v2 only accepts linter names (lowercase) in //nolint directives, not individual check IDs like SA1019
	plugin.Register(disagg.PdProfileHandlerType, disagg.PdProfileHandlerFactory) //nolint:staticcheck // intentional: keep backward compatibility (SA1019)
	plugin.Register(preciseprefixcache.PrecisePrefixCachePluginType, preciseprefixcache.PluginFactory)
	plugin.Register(loadaware.LoadAwareType, loadaware.Factory)
	plugin.Register(sessionaffinity.SessionAffinityType, sessionaffinity.Factory)
	plugin.Register(activerequest.ActiveRequestType, activerequest.Factory)
	plugin.Register(nohitlru.NoHitLRUType, nohitlru.Factory)
	plugin.Register(srcmodels.ModelsDataSourceType, srcmodels.ModelDataSourceFactory)
	plugin.Register(extmodels.ModelsExtractorType, extmodels.ModelServerExtractorFactory)
	plugin.RegisterAsDefaultProducer(mmproducer.ProducerType, mmproducer.Factory, mmproducer.ProducedKey)
	// pd decider plugins
	plugin.Register(disagg.PrefixBasedPDDeciderPluginType, disagg.PrefixBasedPDDeciderPluginFactory)
	plugin.Register(disagg.AlwaysDisaggPDDeciderPluginType, disagg.AlwaysDisaggPDDeciderPluginFactory)
	plugin.Register(tokenizer.PluginType, tokenizer.PluginFactory)
	plugin.RegisterAsDefaultProducer(inflightload.InFlightLoadProducerType, inflightload.InFlightLoadProducerFactory, attrconcurrency.InFlightLoadKey)
	// Legacy alias - existing YAML configs using "tokenizer" continue to work, with a deprecation warning.
	plugin.Register(tokenizer.LegacyPluginType, tokenizer.LegacyPluginFactory) //nolint:staticcheck // intentional: keep backward compatibility (SA1019)
	// ep decider plugins
	plugin.Register(disagg.AlwaysDisaggMulimodalPluginType, disagg.AlwaysDisaggMulimodalDeciderPluginFactory)
	plugin.Register(contextlengthaware.ContextLengthAwareType, contextlengthaware.Factory)
	plugin.Register(mmcacheaffinity.Type, mmcacheaffinity.Factory)
}
