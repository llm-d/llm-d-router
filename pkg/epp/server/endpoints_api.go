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

package server

import (
	"cmp"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
)

// EndpointsAPIPath serves a read-only snapshot of the endpoints the datastore
// currently tracks. Registered only when --enable-endpoints-api is set.
const EndpointsAPIPath = "/v1/endpoints"

// emptyLabels stands in for nil labels; json.Marshal only reads it.
var emptyLabels = map[string]string{}

type endpointsAPIResponse struct {
	Endpoints []endpointsAPIEntry `json:"endpoints"`
}

// endpointsAPIEntry is one datastore endpoint. A pod serving several target
// ports appears once per port; port and rank tell the entries apart. A
// data-parallel pod appears once per rank on the same port, and rank is then
// the data-parallel rank, not a target-port index.
type endpointsAPIEntry struct {
	Name      string            `json:"name"`
	Namespace string            `json:"namespace"`
	Address   string            `json:"address"`
	Port      string            `json:"port"`
	Rank      int               `json:"rank"`
	Labels    map[string]string `json:"labels"`
}

// NewEndpointsAPIHandler returns a GET-only handler listing the datastore's
// endpoints with their labels. It reads the in-memory snapshot, so a request
// never reaches the Kubernetes API. In Kubernetes discovery the datastore
// only holds Ready pods that match the pool selector. Until ready reports
// true (discovery's initial sync is done) it answers 503, so clients never
// read a not-yet-populated datastore as an empty pool.
func NewEndpointsAPIHandler(ds datastore.Datastore, ready func() bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !ready() {
			http.Error(w, "endpoint discovery has not completed its initial sync", http.StatusServiceUnavailable)
			return
		}

		endpoints := ds.PodList(datastore.AllPodsPredicate)
		resp := endpointsAPIResponse{Endpoints: make([]endpointsAPIEntry, 0, len(endpoints))}
		for _, ep := range endpoints {
			md := ep.GetMetadata()
			if md == nil {
				continue
			}
			labels := md.Labels
			if labels == nil {
				labels = emptyLabels // clients index labels directly; never send null
			}
			resp.Endpoints = append(resp.Endpoints, endpointsAPIEntry{
				Name:      md.Name,
				Namespace: md.ID.Namespace,
				Address:   md.Address,
				Port:      md.Port,
				Rank:      md.RankIndex,
				Labels:    labels,
			})
		}
		slices.SortFunc(resp.Endpoints, func(a, b endpointsAPIEntry) int {
			return cmp.Or(cmp.Compare(a.Namespace, b.Namespace), cmp.Compare(a.Name, b.Name), cmp.Compare(a.Rank, b.Rank))
		})

		payload, err := json.Marshal(resp)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to encode endpoints: %v", err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	})
}
