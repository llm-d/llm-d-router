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

package plugin

import "fmt"

// DataKey uniquely identifies the data for data producer/consumer.
type DataKey struct {
	dataType     string
	producerName string
}

// NewDataKey creates a new DataKey.
// The defaultProducerName is passed as the initial producerName.
func NewDataKey(dataType, defaultProducerName string) DataKey {
	return DataKey{
		dataType:     dataType,
		producerName: defaultProducerName,
	}
}

// WithNonEmptyProducerName returns a copy of the key with the specified producer name
// if the name is not empty, otherwise returns the key unchanged.
func (dk DataKey) WithNonEmptyProducerName(name string) DataKey {
	if name != "" {
		dk.producerName = name
	}
	return dk
}

// String serializes the key to "DataType/ProducerName".
func (dk DataKey) String() string {
	return fmt.Sprintf("%s/%s", dk.dataType, dk.producerName)
}
