# Output-Length Sketch

An online, CPU-only estimator of a request's output length, used by the
`inflight-load-producer` as a more precise alternative to the `outlen-bucket`
signal rules (`outputEstimator: sketch`). It needs no training and keeps adapting
to the live workload.

## Pipeline

Per request, before scheduling:

1. **Window** the prompt to a bounded set of structure-bearing turns (system, first
   user, last user, last tool), taking head+tail of any that overflow a character
   budget. Keeps the length-predictive cues, bounds the cost.
2. **Embed** the window with a static table-lookup embedding (model2vec
   potion-base-8M): WordPiece-tokenize, look up each token's row, mean-pool,
   L2-normalize. No neural forward pass, no model server.
3. **Key** the embedding by its nearest of K online k-means centroids (the
   AdaptiveKey), folded with a signature of the request signals (model, reasoning,
   tools, output cap, ...), so every signal combination gets its own histogram.
4. **Read** the key's count-decaying histogram of observed output lengths at a high
   quantile, mapping the chosen output-size bin to a representative token magnitude.
   A key below the warmup count abstains, so the caller falls back to the static
   estimate until the key warms up.

At end-of-stream the observed completion length folds back in: it nudges the
AdaptiveKey centroid toward the prompt embedding and updates the key's histogram.

## AdaptiveKey

The centroids never hard freeze. A first batch of observations is fit with Lloyd's
algorithm (accepted only when enough centroids have members); every later
observation nudges its nearest centroid by a cumulative-mean step with learning rate
`1/(n+1)`, so the step decays on its own and the centroids track slow workload drift.
On stationary traffic this matches a one-shot frozen fit, so adapting costs nothing
at rest.

## Output-size bins

Data-driven (distribution valleys): S `<500`, M `500-2000`, L `2000-7000`,
XL `7000-20000`, XXL `20000+`. The prediction maps the chosen bin to its
representative token magnitude.

## Model asset

The embedding model (`vocab.txt` + `model.safetensors`, ~30 MB float32) is mounted
at the `sketch.modelDir` path and loaded once at startup; it is read-only on the hot
path.
