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

package requestcontrol

import (
	"sort"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
)

// NewConfig creates a new Config object and returns its pointer.
func NewConfig() *Config {
	return &Config{
		requestHeaderPlugins:     []fwkrc.RequestHeaderProcessor{},
		screeners:                []fwkrc.Screener{},
		admissionPlugins:         []fwkrc.Admitter{},
		dataProducerPlugins:      []fwkrc.DataProducer{},
		preRequestPlugins:        []fwkrc.PreRequest{},
		responseReceivedPlugins:  []fwkrc.ResponseHeaderProcessor{},
		responseStreamingPlugins: []fwkrc.ResponseBodyProcessor{},
	}
}

// Config provides a configuration for the requestcontrol plugins.
type Config struct {
	requestHeaderPlugins     []fwkrc.RequestHeaderProcessor
	screeners                []fwkrc.Screener
	admissionPlugins         []fwkrc.Admitter
	dataProducerPlugins      []fwkrc.DataProducer
	preRequestPlugins        []fwkrc.PreRequest
	responseReceivedPlugins  []fwkrc.ResponseHeaderProcessor
	responseStreamingPlugins []fwkrc.ResponseBodyProcessor
}

// WithRequestHeaderPlugins sets the given plugins as the RequestHeaderProcessor plugins.
func (c *Config) WithRequestHeaderPlugins(plugins ...fwkrc.RequestHeaderProcessor) *Config {
	c.requestHeaderPlugins = plugins
	return c
}

// WithScreeners sets the screeners that narrow candidates before request data
// production and scheduling.
func (c *Config) WithScreeners(screeners ...fwkrc.Screener) *Config {
	c.screeners = screeners
	return c
}

// WithPreRequestPlugins sets the given plugins as the PreRequest plugins.
// If the Config has PreRequest plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithPreRequestPlugins(plugins ...fwkrc.PreRequest) *Config {
	c.preRequestPlugins = plugins
	return c
}

// WithResponseReceivedPlugins sets the given plugins as the ResponseReceived plugins.
// If the Config has ResponseReceived plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseReceivedPlugins(plugins ...fwkrc.ResponseHeaderProcessor) *Config {
	c.responseReceivedPlugins = plugins
	return c
}

// WithResponseStreamingPlugins sets the given plugins as the ResponseStreaming plugins.
// If the Config has ResponseStreaming plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseStreamingPlugins(plugins ...fwkrc.ResponseBodyProcessor) *Config {
	c.responseStreamingPlugins = plugins
	return c
}

// WithDataProducerPlugins sets the given plugins as the DataProducer plugins.
func (c *Config) WithDataProducerPlugins(plugins ...fwkrc.DataProducer) *Config {
	c.dataProducerPlugins = plugins
	return c
}

// WithAdmissionPlugins sets the given plugins as the Admit plugins.
func (c *Config) WithAdmissionPlugins(plugins ...fwkrc.Admitter) *Config {
	c.admissionPlugins = plugins
	return c
}

// AddPlugins adds the given plugins to the Config.
// The type of each plugin is checked and added to the corresponding list of plugins in the Config.
// If a plugin implements multiple plugin interfaces, it will be added to each corresponding list.
func (c *Config) AddPlugins(pluginObjects ...plugin.Plugin) {
	for _, plugin := range pluginObjects {
		if requestHeaderProcessor, ok := plugin.(fwkrc.RequestHeaderProcessor); ok {
			c.requestHeaderPlugins = append(c.requestHeaderPlugins, requestHeaderProcessor)
		}
		if screener, ok := plugin.(fwkrc.Screener); ok {
			c.screeners = append(c.screeners, screener)
		}
		if preRequestPlugin, ok := plugin.(fwkrc.PreRequest); ok {
			c.preRequestPlugins = append(c.preRequestPlugins, preRequestPlugin)
		}
		if responseReceivedPlugin, ok := plugin.(fwkrc.ResponseHeaderProcessor); ok {
			c.responseReceivedPlugins = append(c.responseReceivedPlugins, responseReceivedPlugin)
		}
		if responseStreamingPlugin, ok := plugin.(fwkrc.ResponseBodyProcessor); ok {
			c.responseStreamingPlugins = append(c.responseStreamingPlugins, responseStreamingPlugin)
		}
		if dataProducerPlugin, ok := plugin.(fwkrc.DataProducer); ok {
			c.dataProducerPlugins = append(c.dataProducerPlugins, dataProducerPlugin)
		}
		if admissionPlugin, ok := plugin.(fwkrc.Admitter); ok {
			c.admissionPlugins = append(c.admissionPlugins, admissionPlugin)
		}
	}
}

// OrderPlugins reorders every extension point in the Config to the given sorted
// plugin names, so that a plugin runs after the plugins whose data it consumes.
// The names come from the data-dependency DAG, which only ranks producers and
// consumers; plugins it does not rank keep running, ordered by name after the
// ranked ones.
func (c *Config) OrderPlugins(sortedPluginNames []string) {
	rank := make(map[string]int, len(sortedPluginNames))
	for i, name := range sortedPluginNames {
		rank[name] = i
	}

	c.requestHeaderPlugins = orderByName(c.requestHeaderPlugins, rank)
	c.screeners = orderByName(c.screeners, rank)
	c.admissionPlugins = orderByName(c.admissionPlugins, rank)
	c.dataProducerPlugins = orderByName(c.dataProducerPlugins, rank)
	c.preRequestPlugins = orderByName(c.preRequestPlugins, rank)
	c.responseReceivedPlugins = orderByName(c.responseReceivedPlugins, rank)
	c.responseStreamingPlugins = orderByName(c.responseStreamingPlugins, rank)
}

// orderByName returns the plugins ordered by their rank, followed by the unranked
// ones sorted by name. Sorting the unranked plugins rather than keeping their
// original order matters: they reach the Config through a map iteration, so their
// incoming order differs between runs.
func orderByName[T plugin.Plugin](plugins []T, rank map[string]int) []T {
	ordered := make([]T, len(plugins))
	copy(ordered, plugins)

	sort.SliceStable(ordered, func(i, j int) bool {
		iName, jName := ordered[i].TypedName().String(), ordered[j].TypedName().String()
		iRank, iRanked := rank[iName]
		jRank, jRanked := rank[jName]
		if iRanked != jRanked {
			return iRanked
		}
		if !iRanked {
			return iName < jName
		}
		return iRank < jRank
	})

	return ordered
}
