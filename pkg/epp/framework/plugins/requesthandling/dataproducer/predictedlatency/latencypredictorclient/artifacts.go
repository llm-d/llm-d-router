/*
Copyright 2026 The llm-d Authors.

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

package latencypredictorclient

import (
	"bytes"
	"encoding/json"
	"io"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol"
	"google.golang.org/protobuf/types/known/structpb"
)

func (info *ModelInfo) UnmarshalJSON(data []byte) error {
	message := &protocol.ModelInfo{}
	if err := protocol.Unmarshal(data, message); err != nil {
		return err
	}
	preserveEmptyMap(data, "model_status", &message.ModelStatus)
	*info = ModelInfo{ModelType: message.GetModelType(), ObjectiveType: message.GetObjectiveType(), ModelStatus: message.GetModelStatus(), Quantile: message.GetQuantile()}
	return nil
}

func (coefficients *ModelCoefficients) UnmarshalJSON(data []byte) error {
	message := &protocol.ModelCoefficients{}
	if err := protocol.Unmarshal(data, message); err != nil {
		return err
	}
	preserveEmptyMap(data, "ttft_coefficients", &message.TtftCoefficients)
	preserveEmptyMap(data, "tpot_coefficients", &message.TpotCoefficients)
	*coefficients = ModelCoefficients{TTFTIntercept: message.GetTtftIntercept(), TTFTCoeffs: message.GetTtftCoefficients(), TPOTIntercept: message.GetTpotIntercept(), TPOTCoeffs: message.GetTpotCoefficients()}
	return nil
}

func readTrees(reader io.Reader) ([]any, error) {
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return nil, nil
	}
	trees := &structpb.ListValue{}
	if err := protocol.Unmarshal(data, trees); err != nil {
		return nil, err
	}
	return trees.AsSlice(), nil
}

// Protobuf maps have no presence; the public JSON facade distinguishes {} from null.
func preserveEmptyMap[V any](data []byte, name string, destination *map[string]V) {
	if *destination != nil {
		return
	}
	var object map[string]json.RawMessage
	if json.Unmarshal(data, &object) != nil {
		return
	}
	if raw, ok := object[name]; ok && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		*destination = make(map[string]V)
	}
}
