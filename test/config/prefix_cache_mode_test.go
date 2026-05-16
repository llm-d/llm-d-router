package config_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-router/pkg/epp/config/loader"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins"
	preciseprefixcache "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/preciseprefixcache"
	testutils "github.com/llm-d/llm-d-router/test/utils"
	igwtestutils "github.com/llm-d/llm-d-router/test/utils/igw"
)

func TestScorer(t *testing.T) {
	tests := []struct {
		name       string
		pluginName string
		configText string
	}{
		{
			name:       "precise prefix cache scorer",
			pluginName: "precisePrefixCache",
			configText: `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: precisePrefixCache
  type: precise-prefix-cache-scorer
  parameters:
    kvEventsConfig:
      zmqEndpoint: "tcp://localhost:5557"
- name: profileHandler
  type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: precisePrefixCache
`,
		},
	}
	ctx := testutils.NewTestContext(t)
	// Register llm-d-inference-scheduler plugins
	plugins.RegisterAllPlugins()

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_ = os.Setenv("HF_TOKEN", "dummy_token") // needed for cache_tracking
			rawConfig, _, err := loader.LoadRawConfig([]byte(test.configText), logr.Discard())
			if err != nil {
				t.Fatalf("unexpected error from LoadConfigPhaseOne: %v", err)
			}
			handle := igwtestutils.NewTestHandle(ctx)
			_, err = loader.InstantiateAndConfigure(rawConfig, handle, logr.Discard())
			if err != nil {
				t.Fatalf("unexpected error from LoadConfigPhaseTwo: %v", err)
			}
			fmt.Println("all plugins", handle.GetAllPluginsWithNames())

			_, err = fwkplugin.PluginByType[*preciseprefixcache.Scorer](handle, test.pluginName)
			if err != nil {
				t.Fatalf("expected Scorer, but got error: %v", err)
			}
		})
	}
}
