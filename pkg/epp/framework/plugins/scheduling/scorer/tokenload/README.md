# Token Load Scorer Plugin

**Type:** `token-load-scorer`

This plugin scores candidate endpoints based on the estimated number of in-flight tokens currently being processed by each model server.

## What it does

For each scheduling cycle, the plugin reads the `InFlightLoad` attribute from each endpoint and computes a normalized score based on the total estimated tokens in flight:

$$
\text{score(endpoint)} = 1.0 - \min\left(1.0, \frac{\text{tokens(endpoint)}}{\text{queueThresholdTokens}}\right)
$$

where tokens counts each multimodal (image, audio, video) token `nonTextTokenWeight` times:

$$
\text{tokens(endpoint)} = \text{textTokens} + \text{nonTextTokenWeight} \times \text{nonTextTokens}
$$

So:
- 0 tokens in flight → score `1.0`
- `queueThresholdTokens` or more tokens → score `0.0`
- others are linearly scaled between them

This scoring mechanism provides a more granular signal than simple request concurrency by accounting for the size of the prompts being processed.

## Scheduling intent

The scorer returns category `Distribution`, helping spread requests away from endpoints that are processing a high volume of tokens, which often correlates with higher latency and resource utilization.

## Inputs consumed

The plugin consumes:
- `attrconcurrency.InFlightLoadDataKey` (`*attrconcurrency.InFlightLoad`)

## Configuration

The scorer supports the following runtime parameters:

- `queueThresholdTokens` (integer, default: 4194304): The maximum number of in-flight tokens used for score normalization. Endpoints exceeding this threshold will receive a score of `0.0`. The default (4Mi tokens) is equivalent to 128 requests with an average size of 32K tokens.
- `nonTextTokenWeight` (float, default: 1.0): How much one multimodal token counts toward the load, relative to a text token. Image, audio, and video tokens are heavier to process than text, so a value above 1.0 keeps a mixed load from looking lighter than it is. The default scores every token alike; non-positive values fall back to it.

**Configuration Example:**
```yaml
plugins:
  - type: token-load-scorer
    name: token-load
    parameters:
      queueThresholdTokens: 4194304
      nonTextTokenWeight: 1.0 # raise to count multimodal tokens as heavier than text
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: token-load
        weight: 1
```
