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

// Package framework is the root of the shared test framework tree. Its
// subpackages provide reusable test helpers layered by what they are
// allowed to import:
//
//   - R1: test/framework/{net,context,k8s} import only the standard
//     library, Kubernetes API/machinery packages, and logr. They never
//     import pkg/.
//   - R2: test/framework/gaie additionally imports GAIE/apix API types
//     and pkg/common/routing. It never imports pkg/epp.
//   - R3: test/framework/epp imports R1/R2 packages, leaf pkg/epp
//     packages (metadata), and pkg/common. It never imports
//     pkg/epp/server or cmd/.
//   - R4: test/framework/epp/harness may import anything; it is imported
//     only by test/integration/epp and e2e suites, never by pkg/ tests.
//   - R5: non-test code under pkg/ must not import test/framework.
package framework
