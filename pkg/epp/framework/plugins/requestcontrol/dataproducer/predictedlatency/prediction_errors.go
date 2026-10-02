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

package predictedlatency

import "errors"

const (
	predictionFailureReasonRequestError   = "request_error"
	predictionFailureReasonPredictorError = "predictor_error"
	predictionFailureReasonNilResponse    = "nil_response"
	predictionFailureReasonLengthMismatch = "length_mismatch"
)

type predictionFailure struct {
	reason string
	err    error
}

func (e *predictionFailure) Error() string {
	return e.err.Error()
}

func (e *predictionFailure) Unwrap() error {
	return e.err
}

func newPredictionFailure(reason string, err error) error {
	if err == nil {
		err = errors.New("prediction failed")
	}
	return &predictionFailure{reason: reason, err: err}
}

func predictionFailureReasonForError(err error) string {
	var failure *predictionFailure
	if errors.As(err, &failure) {
		return failure.reason
	}
	return predictionFailureReasonPredictorError
}
