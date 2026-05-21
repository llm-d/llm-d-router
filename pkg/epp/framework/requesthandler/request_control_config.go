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

package requesthandler

import (
	fwkrhapi "github.com/llm-d/llm-d-router/pkg/epp/framework/requesthandler/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// NewConfig creates a new Config object and returns its pointer.
func NewConfig() *Config {
	return &Config{
		admissionPlugins:         []fwkrhapi.Admitter{},
		dataProducerPlugins:      []fwkrhapi.DataProducer{},
		preRequestPlugins:        []fwkrhapi.PreRequest{},
		responseReceivedPlugins:  []fwkrhapi.ResponseHeaderProcessor{},
		responseStreamingPlugins: []fwkrhapi.ResponseBodyProcessor{},
	}
}

// Config provides a configuration for the requesthandler plugins.
type Config struct {
	admissionPlugins         []fwkrhapi.Admitter
	dataProducerPlugins      []fwkrhapi.DataProducer
	preRequestPlugins        []fwkrhapi.PreRequest
	responseReceivedPlugins  []fwkrhapi.ResponseHeaderProcessor
	responseStreamingPlugins []fwkrhapi.ResponseBodyProcessor
}

// WithPreRequestPlugins sets the given plugins as the fwkrhapi.PreRequest plugins.
// If the Config has fwkrhapi.PreRequest plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithPreRequestPlugins(plugins ...fwkrhapi.PreRequest) *Config {
	c.preRequestPlugins = plugins
	return c
}

// WithResponseReceivedPlugins sets the given plugins as the ResponseReceived plugins.
// If the Config has ResponseReceived plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseReceivedPlugins(plugins ...fwkrhapi.ResponseHeaderProcessor) *Config {
	c.responseReceivedPlugins = plugins
	return c
}

// WithResponseStreamingPlugins sets the given plugins as the ResponseStreaming plugins.
// If the Config has ResponseStreaming plugins already, this call replaces the existing plugins with the given ones.
func (c *Config) WithResponseStreamingPlugins(plugins ...fwkrhapi.ResponseBodyProcessor) *Config {
	c.responseStreamingPlugins = plugins
	return c
}

// WithDataProducerPlugins sets the given plugins as the fwkrhapi.DataProducer plugins.
func (c *Config) WithDataProducerPlugins(plugins ...fwkrhapi.DataProducer) *Config {
	c.dataProducerPlugins = plugins
	return c
}

// WithAdmissionPlugins sets the given plugins as the AdmitRequest plugins.
func (c *Config) WithAdmissionPlugins(plugins ...fwkrhapi.Admitter) *Config {
	c.admissionPlugins = plugins
	return c
}

// AddPlugins adds the given plugins to the Config.
// The type of each plugin is checked and added to the corresponding list of plugins in the Config.
// If a plugin implements multiple plugin interfaces, it will be added to each corresponding list.
func (c *Config) AddPlugins(pluginObjects ...plugin.Plugin) {
	for _, plugin := range pluginObjects {
		if preRequestPlugin, ok := plugin.(fwkrhapi.PreRequest); ok {
			c.preRequestPlugins = append(c.preRequestPlugins, preRequestPlugin)
		}
		if responseReceivedPlugin, ok := plugin.(fwkrhapi.ResponseHeaderProcessor); ok {
			c.responseReceivedPlugins = append(c.responseReceivedPlugins, responseReceivedPlugin)
		}
		if responseStreamingPlugin, ok := plugin.(fwkrhapi.ResponseBodyProcessor); ok {
			c.responseStreamingPlugins = append(c.responseStreamingPlugins, responseStreamingPlugin)
		}
		if dataProducerPlugin, ok := plugin.(fwkrhapi.DataProducer); ok {
			c.dataProducerPlugins = append(c.dataProducerPlugins, dataProducerPlugin)
		}
		if admissionPlugin, ok := plugin.(fwkrhapi.Admitter); ok {
			c.admissionPlugins = append(c.admissionPlugins, admissionPlugin)
		}
	}
}

// OrderDataProducerPlugins reorders the fwkrhapi.DataProducer plugins in the Config based on the given sorted plugin names.
func (c *Config) OrderDataProducerPlugins(sortedPluginNames []string) {
	sortedPlugins := make([]fwkrhapi.DataProducer, 0, len(sortedPluginNames))
	nameToPlugin := make(map[string]fwkrhapi.DataProducer)
	for _, plugin := range c.dataProducerPlugins {
		nameToPlugin[plugin.TypedName().String()] = plugin
	}
	for _, name := range sortedPluginNames {
		if plugin, ok := nameToPlugin[name]; ok {
			sortedPlugins = append(sortedPlugins, plugin)
		}
	}
	c.dataProducerPlugins = sortedPlugins
}
