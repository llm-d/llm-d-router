# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from inference_perf.metrics.request_collector import RequestMetricCollector
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig, MultiLoRAConfig
from inference_perf.apis import (
    InferenceAPIData,
    InferenceInfo,
    RequestLifecycleMetric,
    ErrorResponseInfo,
    StreamedResponseMetrics,
)
from inference_perf.apis.anthropic_messages import ANTHROPIC_VERSION, parse_anthropic_content
from inference_perf.apis.streaming_parser import StreamInterruptedError
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.utils import CustomTokenizer
from .base import ModelServerClient, ModelServerClientSession
from .metrics import Metric, BaseMetrics
from .otel_instrumentation import get_otel_instrumentation
from typing import Iterator, List, Optional, Any, Dict, Tuple
import aiohttp
import asyncio
import random
from asyncio import sleep
import json
import time
import logging
import requests
import ssl


logger = logging.getLogger(__name__)


class OpenAIMetrics(BaseMetrics):
    def __init__(
        self,
        filters: List[str],
        prompt_tokens: Metric[Any],
        output_tokens: Metric[Any],
        requests: Metric[Any],
        request_latency: Metric[Any],
        queue_length: Metric[Any],
        time_per_output_token: Metric[Any],
        custom_metrics: Optional[Dict[str, Metric[Any]]] = None,
    ) -> None:
        super().__init__(filters, custom_metrics)
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.requests = requests
        self.request_latency = request_latency
        self.queue_length = queue_length
        self.time_per_output_token = time_per_output_token

    def _iter_metrics(self) -> Iterator[Tuple[str, Metric[Any]]]:
        yield "prompt_tokens", self.prompt_tokens
        yield "output_tokens", self.output_tokens
        yield "requests", self.requests
        yield "request_latency", self.request_latency
        yield "queue_length", self.queue_length
        yield "time_per_output_token", self.time_per_output_token
        yield from super()._iter_metrics()


class openAIModelServerClient(ModelServerClient):
    _session: "Optional[openAIModelServerClientSession]" = None
    _session_lock = asyncio.Lock()

    def __init__(
        self,
        metrics_collector: RequestMetricCollector,
        api_config: APIConfig,
        uri: str,
        model_name: Optional[str],
        tokenizer_config: Optional[CustomTokenizerConfig],
        max_tcp_connections: int,
        additional_filters: List[str],
        ignore_eos: bool = True,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        cert_path: Optional[str] = None,
        key_path: Optional[str] = None,
        lora_config: Optional[List[MultiLoRAConfig]] = None,
        request_retries: int = 0,
        request_retry_backoff_sec: float = 0.5,
    ) -> None:
        super().__init__(
            api_config,
            timeout,
            request_retries=request_retries,
            request_retry_backoff_sec=request_retry_backoff_sec,
        )
        self.uri = uri
        self.max_completion_tokens = 30  # default to use when not set at the request level
        self.ignore_eos = ignore_eos
        self.metrics_collector = metrics_collector
        self.max_tcp_connections = max_tcp_connections
        self.additional_filters = additional_filters
        self.api_key = api_key
        self.cert_path = cert_path
        self.key_path = key_path
        self.lora_config = lora_config

        # Initialize OTEL instrumentation (configured via environment variables)
        self.otel = get_otel_instrumentation()

        if model_name is None:
            supported_models = self.get_supported_models()
            if not supported_models:
                logger.error("No supported models found")
                raise Exception("openAI client init failed, no model_name could be found")
            inferred_id = supported_models[0].get("id")
            if not isinstance(inferred_id, str):
                raise Exception(f"openAI client init failed, model entry has no string 'id': {supported_models[0]}")
            self.model_name: str = inferred_id
            logger.info(f"Inferred model {self.model_name}")
            if len(supported_models) > 1:
                logger.warning(f"More than one supported model found {supported_models}, selecting {self.model_name}")
        else:
            self.model_name = model_name

        if self.lora_config is not None:
            supported_models = self.get_supported_models()
            supported_model_names = set()
            for model in supported_models:
                supported_model_names.add(model.get("id"))
            lora_adapters = [config.name for config in self.lora_config]
            for adapter in lora_adapters:
                if adapter not in supported_model_names:
                    raise ValueError(f"LoRA adapter {adapter} not found in model server's available models")

        if tokenizer_config and not tokenizer_config.pretrained_model_name_or_path:
            tokenizer_config.pretrained_model_name_or_path = self.model_name
        elif not tokenizer_config:
            tokenizer_config = CustomTokenizerConfig(pretrained_model_name_or_path=self.model_name)
        self.tokenizer = CustomTokenizer(tokenizer_config)

    def new_session(self) -> "ModelServerClientSession":
        return openAIModelServerClientSession(self)

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        """
        Create an internal client session if not already, then use that to
        process the request.
        """
        session: openAIModelServerClientSession
        # ensure session is only created once.
        async with self._session_lock:
            if self._session is None:
                self._session = openAIModelServerClientSession(self)
            session = self._session
        await session.process_request(data, stage_id, scheduled_time, lora_adapter)

    async def close(self) -> None:
        """Close the internal session created by process_request, if any."""
        if self._session is not None:
            await self._session.close()
            self._session = None

        # Shutdown OTEL instrumentation to flush pending spans
        if self.otel:
            self.otel.shutdown()

    def get_supported_apis(self) -> List[APIType]:
        return []

    @abstractmethod
    def get_prometheus_metric_metadata(self) -> OpenAIMetrics:
        raise NotImplementedError

    def get_supported_models(self) -> List[dict[str, Any]]:
        try:
            response = requests.get(f"{self.uri}/v1/models")
            response.raise_for_status()
            data = response.json()
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            else:
                return []
        except Exception as e:
            logger.error(f"Got exception retrieving supported models {e}")
            return []


def _update_headers_case_insensitive(target: dict[str, str], source: dict[str, str]) -> None:
    for k, v in source.items():
        k_lower = k.lower()
        matching_keys = [exist_k for exist_k in target.keys() if exist_k.lower() == k_lower]
        for mk in matching_keys:
            del target[mk]
        target[k] = v


def is_retryable_transport_error(exc: BaseException) -> bool:
    """True for a connection fault raised before response headers were obtained.

    ``ClientConnectionError`` is exactly that family: a connection refused, reset, or
    closed while idle in the pool. ``ClientPayloadError`` sits deliberately outside it --
    a body that broke mid-transfer is past the point where a response was established, so
    retrying it could mix or discard token-level measurements and duplicate work the server
    already did.

    The boundary is the *client's* view, and it is narrower than "the server did no
    work". No response was established, so no TTFT or partial ITLs exist to be
    contaminated -- but the request bytes may already have been sent, and the server may
    have received or begun processing the attempt. A retry therefore avoids mixing partial
    client-side response measurements across attempts, while still being capable of
    duplicating work on the endpoint -- which is why it is bounded (``request_retries``)
    and backed off rather than unconditional.

    Two families are excluded despite inheriting from ``ClientConnectionError``:

    - Timeouts (``ServerTimeoutError``, ``SocketTimeoutError``). The request may have
      reached the model, and ``request_timeout`` applies per attempt, so re-sending
      multiplies the deadline.
    - Deterministic TLS failures. A rejected certificate or a mismatched fingerprint is a
      configuration error, not a transient one: every attempt fails identically, so
      retrying only spends the backoff. ``ClientSSLError`` covers both connector variants,
      and ``ClientConnectorCertificateError`` in particular also inherits from
      ``ClientOSError`` -- the class we retry for connection resets -- so it has to be
      excluded by name rather than left to the family check.

    Note this excludes TLS *verification* failures, not connectivity ones:
    ``ClientConnectorError`` (connection refused, DNS failure) stays retryable, since those
    do recover.
    """
    if isinstance(exc, (asyncio.TimeoutError, aiohttp.ClientSSLError, aiohttp.ServerFingerprintMismatch)):
        return False
    return isinstance(exc, aiohttp.ClientConnectionError)


class openAIModelServerClientSession(ModelServerClientSession):
    client: openAIModelServerClient

    def __init__(self, client: openAIModelServerClient):
        timeout = aiohttp.ClientTimeout(total=client.timeout) if client.timeout else aiohttp.helpers.sentinel
        connector = None
        if client.cert_path and client.key_path:
            ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)  # Use system trust store
            ssl_context.load_cert_chain(certfile=client.cert_path, keyfile=client.key_path)
            connector = aiohttp.TCPConnector(limit=client.max_tcp_connections, ssl=ssl_context)
        else:
            connector = aiohttp.TCPConnector(limit=client.max_tcp_connections)

        self.client = client
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        # Server-assigned session tokens keyed by session identity, captured from
        # the response header named by api_config.session_token_header_key.
        # Sessions are pinned to a worker (preferred_worker_id), so a per-worker
        # store sees every request of its sessions.
        self._session_tokens: dict[str, str] = {}

    def _get_session_otel_context(self, data: InferenceAPIData) -> Optional[Dict[str, str]]:
        """Get session OTEL context if available (for OTel trace replay)."""

        if hasattr(data, "otel_context") and data.otel_context is not None:
            return data.otel_context

        return None

    def _record_otel_metrics(
        self,
        span: Any,
        data: InferenceAPIData,
        response: Optional[aiohttp.ClientResponse],
        info: Optional[InferenceInfo],
        response_content: str,
        error: Optional[ErrorResponseInfo],
        start_time: float,
        end_time: float,
        retry_wasted_sec: Optional[float] = None,
    ) -> None:
        """Record OTEL metrics for the request.

        ``start_time`` is the logical request's dispatch time, so ``total_latency`` matches
        the span's own wall duration -- the span opens outside the attempt loop and
        therefore already covers every attempt plus backoff. ``retry_wasted_sec`` is the
        share of that duration lost to attempts that failed, computed by the caller so this
        span and the request metric report one number rather than two derivations of it;
        it is None on a request that never retried.
        """
        if not self.client.otel.enabled or span is None:
            return

        if info:
            inner = info.response_metrics
            otel_response_info: Dict[str, Any] = {
                "prompt_tokens": info.request_metrics.text.input_tokens,
                "completion_tokens": inner.output_tokens if inner else 0,
                "total_latency": end_time - start_time,
            }

            # Time lost to failed attempts and backoff. total_latency above still counts
            # it; this says how much of it was waste.
            if retry_wasted_sec is not None:
                otel_response_info["retry_wasted_sec"] = retry_wasted_sec

            # Calculate TTFT if token times are available (streaming only)
            if isinstance(inner, StreamedResponseMetrics) and inner.output_token_times:
                ttft = inner.output_token_times[0] - start_time
                otel_response_info["time_to_first_token"] = ttft

                # Calculate average TPOT if we have multiple tokens
                if len(inner.output_token_times) > 1:
                    total_decode_time = inner.output_token_times[-1] - inner.output_token_times[0]
                    num_decode_tokens = len(inner.output_token_times) - 1
                    tpot = total_decode_time / num_decode_tokens if num_decode_tokens > 0 else 0
                    otel_response_info["time_per_output_token"] = tpot

            # Add finish reason from extra_info if available
            if "finish_reason" in info.extra_info:
                otel_response_info["finish_reason"] = info.extra_info["finish_reason"]

            # Extract input and output following GenAI semantic conventions
            try:
                # Extract input based on request type
                if hasattr(data, "messages") and data.messages:
                    # Serialize each message, preserving tool_calls when present.
                    input_messages = [msg.to_dict() for msg in data.messages]
                    otel_response_info["input_messages"] = json.dumps(input_messages)

                    # Record tool definitions so they appear in Jaeger alongside the request.
                    if hasattr(data, "tool_definitions") and data.tool_definitions:
                        otel_response_info["tool_definitions"] = json.dumps(data.tool_definitions)
                elif hasattr(data, "prompt"):
                    # Text completion - store as prompt string (gen_ai.prompt)
                    otel_response_info["input_prompt"] = data.prompt

                # Extract output text (gen_ai.output.text)
                if self.client.api_config.type == APIType.AnthropicMessages and response and response.status == 200:
                    if not self.client.api_config.streaming and response_content:
                        response_json = json.loads(response_content)
                        output_text, output_message = parse_anthropic_content(response_json.get("content"))
                        if output_text:
                            otel_response_info["output_text"] = output_text
                        if output_message and output_message.get("tool_calls"):
                            otel_response_info["output_message"] = json.dumps(output_message)
                        if response_json.get("stop_reason"):
                            otel_response_info["finish_reason"] = response_json["stop_reason"]
                elif self.client.api_config.streaming and response_content:
                    stripped_response = response_content.strip()
                    if stripped_response.startswith("data:"):
                        output_parts = []
                        reasoning_parts = []
                        # A streaming tool call arrives as many deltas: id and name on the
                        # first, arguments split across the rest. Accumulate by index so
                        # output_message matches the non-streaming branch's assembled dict.
                        tool_call_chunks: Dict[int, Dict[str, Any]] = {}
                        for line in stripped_response.splitlines():
                            line = line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            payload = line[len("data:") :].strip()
                            if not payload or payload == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            if delta.get("content"):
                                output_parts.append(delta["content"])
                            reasoning_chunk = delta.get("reasoning") or delta.get("reasoning_content")
                            if reasoning_chunk:
                                reasoning_parts.append(reasoning_chunk)
                            for tc_chunk in delta.get("tool_calls") or []:
                                idx = tc_chunk.get("index", 0)
                                if idx not in tool_call_chunks:
                                    tool_call_chunks[idx] = {
                                        "id": tc_chunk.get("id", ""),
                                        "type": tc_chunk.get("type", "function"),
                                        "function": {"name": "", "arguments": ""},
                                    }
                                tc_fn = tc_chunk.get("function") or {}
                                if tc_fn.get("name"):
                                    tool_call_chunks[idx]["function"]["name"] += tc_fn["name"]
                                if tc_fn.get("arguments"):
                                    tool_call_chunks[idx]["function"]["arguments"] += tc_fn["arguments"]
                                if tc_chunk.get("id"):
                                    tool_call_chunks[idx]["id"] = tc_chunk["id"]

                        streamed_text = "".join(output_parts)
                        otel_response_info["output_text"] = streamed_text
                        if reasoning_parts:
                            otel_response_info["reasoning_text"] = "".join(reasoning_parts)
                        if tool_call_chunks:
                            # Same shape as the non-streaming branch: one assistant message dict.
                            assembled_message: Dict[str, Any] = {
                                "role": "assistant",
                                "tool_calls": [tool_call_chunks[i] for i in sorted(tool_call_chunks)],
                            }
                            if streamed_text:
                                assembled_message["content"] = streamed_text
                            otel_response_info["output_message"] = json.dumps(assembled_message)
                elif response and response.status == 200 and response_content:
                    response_json = json.loads(response_content)
                    choices = response_json.get("choices", [])
                    if choices:
                        if "message" in choices[0]:
                            msg_out = choices[0].get("message", {})
                            output_text = msg_out.get("content") or ""
                            reasoning_content = msg_out.get("reasoning") or msg_out.get("reasoning_content")

                            otel_response_info["output_text"] = output_text
                            if reasoning_content:
                                otel_response_info["reasoning_text"] = reasoning_content

                            if msg_out.get("tool_calls") or reasoning_content:
                                otel_response_info["output_message"] = json.dumps(msg_out)
                        elif "text" in choices[0]:
                            output_text = choices[0].get("text", "")
                            otel_response_info["output_text"] = output_text
            except Exception as e:
                logger.warning(f"Failed to extract messages for OTEL: {e}")

            self.client.otel.record_response_metrics(
                span=span,
                response_info=otel_response_info,
                error=error.error_msg if error else None,
            )
        elif error:
            # No `info` to draw response metrics from, but the retry cost is known -- and
            # this is the request that wasted the most, since no attempt answered.
            failed_info: Dict[str, Any] = {}
            if retry_wasted_sec is not None:
                failed_info["retry_wasted_sec"] = retry_wasted_sec
            self.client.otel.record_response_metrics(
                span=span,
                response_info=failed_info or None,
                error=error.error_msg,
            )

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        # Compute effective model name: use LoRA adapter if provided, otherwise use client's model name
        effective_model_name = lora_adapter if lora_adapter else self.client.model_name
        payload = await data.to_request_body(
            effective_model_name=effective_model_name,
            max_tokens=self.client.max_completion_tokens,
            ignore_eos=self.client.ignore_eos,
            streaming=self.client.api_config.streaming,
        )

        # Add response_format for structured output if configured.
        if self.client.api_config.type != APIType.AnthropicMessages and self.client.api_config.response_format:
            payload["response_format"] = self.client.api_config.response_format.to_api_format()

        headers = {"Content-Type": "application/json"}

        if self.client.api_key:
            if self.client.api_config.type == APIType.AnthropicMessages:
                headers["x-api-key"] = self.client.api_key
                headers["anthropic-version"] = ANTHROPIC_VERSION
            else:
                headers["Authorization"] = f"Bearer {self.client.api_key}"

        if self.client.api_config.headers:
            _update_headers_case_insensitive(headers, self.client.api_config.headers)

        if data.headers:
            _update_headers_case_insensitive(headers, data.headers)

        # Trace replay carries session identity as session_id (stamped by the loadgen);
        # conversation_replay and shared_prefix carry it as user_session_id.
        session_id = getattr(data, "session_id", None) or getattr(data, "user_session_id", None)

        if self.client.api_config.session_id_header_key and session_id:
            headers[self.client.api_config.session_id_header_key] = session_id
            # POC patch: weka replay stamps every request of a trace with the
            # same session id, so the EPP cannot tell main-agent turns from
            # subagent fan-out. Graph event ids embed the call id
            # (parent turns: event_NNN_parent_turn_K, subagents:
            # event_NNN_sa_...), so mark only main-agent turns with a session
            # type header; the EPP inter-turn predictor and kv-cache-hint are
            # configured to track exactly that type and ignore the rest.
            event_id = str(getattr(data, "graph_event_id", "") or "")
            if "_parent_turn_" in event_id:
                headers["x-session-type"] = "main"

        session_token_header = self.client.api_config.session_token_header_key
        if session_id and session_token_header:
            session_token = self._session_tokens.get(session_id)
            if session_token:
                headers[session_token_header] = session_token

        request_data = json.dumps(payload)

        # Determine operation name based on API type
        if self.client.api_config.type == APIType.Chat:
            operation_name = "chat.completions"
        elif self.client.api_config.type == APIType.AnthropicMessages:
            operation_name = "messages"
        else:
            operation_name = "completions"

        # Two clocks, deliberately. `start` is the logical request's dispatch time and
        # never moves: the report derives schedule_delay, send_duration, achieved_rate and
        # request latency from it, so re-stamping it would charge a retry's backoff to
        # scheduling and hide time the workload genuinely spent waiting. `attempt_start`
        # tracks the attempt that answered, for the serving-side view of the same request.
        start = time.perf_counter()
        attempt_start = start
        response: Optional[aiohttp.ClientResponse] = None
        info = None
        error = None
        response_content = ""
        caught_exception: Optional[Exception] = None
        retries_attempted = 0
        retries_recovered = False
        # Set once the attempt loop exits; read again after the span closes.
        retry_wasted_sec: Optional[float] = None

        # Get session OTEL context if available (for OTel trace replay)
        parent_context = self._get_session_otel_context(data)

        # Start OTEL tracing
        with self.client.otel.trace_llm_request(
            operation_name=operation_name,
            model_name=effective_model_name,
            request_data=payload,
            parent_context=parent_context,
        ) as span:
            # Attempt loop for transport faults raised before response headers were
            # obtained (see is_retryable_transport_error). Retries stay inside the span and
            # record one RequestLifecycleMetric, so a recovered request is not counted twice.
            max_attempts = 1 + max(0, self.client.request_retries)
            for attempt in range(max_attempts):
                # Only the attempt clock is re-stamped. `start` stays at logical dispatch
                # so end-to-end latency, TTFT and the scheduling metrics keep counting the
                # failed attempt and its backoff -- the workload really did wait for them.
                attempt_start = time.perf_counter()
                response = None
                info = None
                error = None
                response_content = ""
                caught_exception = None
                log_message = ""
                try:
                    async with self.session.post(
                        self.client.uri + data.get_route(), headers=headers, data=request_data
                    ) as resp:
                        response = resp
                        if session_id and session_token_header:
                            received_token = resp.headers.get(session_token_header)
                            if received_token:
                                self._session_tokens[session_id] = received_token
                        try:
                            if self.client.api_config.streaming and response.status == 200:
                                info = await data.process_response(
                                    response=response,
                                    config=self.client.api_config,
                                    tokenizer=self.client.tokenizer,
                                    lora_adapter=lora_adapter,
                                )
                                # pop (not get) to release the raw SSE body from InferenceInfo immediately;
                                # holding it in extra_info for the lifetime of the object causes unbounded
                                # memory growth when many sessions run concurrently.
                                response_content = info.extra_info.pop("raw_response", "") if info else ""
                            else:
                                # Read response body once to avoid double-read issue
                                response_content = await response.text()

                                if response.status == 200:
                                    info = await data.process_response(
                                        response=response,
                                        config=self.client.api_config,
                                        tokenizer=self.client.tokenizer,
                                        lora_adapter=lora_adapter,
                                    )

                            if response.status != 200:
                                # Handle HTTP error responses (status != 200).
                                #
                                # For OTel trace replay, process_failure() is called to:
                                # 1. Mark the session as failed in WorkerSessionTracker
                                # 2. Call registry.record_failure() to unblock dependent events via EventFailedError
                                # 3. Immediately notify the main process via session_completion_queue
                                #
                                # This ensures that if request X fails and request Y depends on X's output,
                                # Y raises EventFailedError and skips rather than hanging indefinitely.
                                #
                                # Note: We call process_failure() for all data types on non-200 responses
                                # to ensure proper state cleanup (e.g. releasing locks in multi-turn chat)
                                # and failure propagation.
                                if response is not None:
                                    error = ErrorResponseInfo(
                                        error_msg=response_content,
                                        error_type=f"HTTP Error {response.status}",
                                    )
                                    exception = Exception(f"{error.error_type}: {error.error_msg}")
                                    info = await data.process_failure(
                                        response=response,
                                        config=self.client.api_config,
                                        tokenizer=self.client.tokenizer,
                                        exception=exception,
                                        lora_adapter=lora_adapter,
                                    )
                        except Exception as read_error:
                            # Handle errors reading response body or streaming.
                            # For 200 responses, process_response() raised (e.g. ClientPayloadError
                            # from a broken SSE stream). Call process_failure() here so that session
                            # locks are released before the context manager exits. Re-raising would
                            # run ClientResponse.__aexit__ on a broken connection, which can raise
                            # a second exception that masks the original and bypasses the outer
                            # aiohttp.ClientError handler.
                            if response is not None and response.status == 200 and not info:
                                caught_exception = read_error
                                # If the stream broke partway, recover the bytes
                                # received so the per-request report shows what the
                                # server actually sent, and report the underlying
                                # exception (e.g. ClientPayloadError) rather than the
                                # StreamInterruptedError wrapper.
                                original_error: Exception = read_error
                                if isinstance(read_error, StreamInterruptedError):
                                    original_error = read_error.original
                                    if read_error.raw_content:
                                        response_content = read_error.raw_content
                                error = ErrorResponseInfo(
                                    error_msg=str(original_error),
                                    error_type=type(original_error).__name__,
                                )
                                info = await data.process_failure(
                                    response=None,
                                    config=self.client.api_config,
                                    tokenizer=self.client.tokenizer,
                                    exception=original_error,
                                    lora_adapter=lora_adapter,
                                )
                            else:
                                if not response_content:
                                    response_content = f"Failed to read response text: {read_error}"
                                raise

                except aiohttp.ClientError as e:
                    caught_exception = e
                    log_message = "Client error during request:"
                    error = ErrorResponseInfo(error_msg=str(e), error_type=type(e).__name__)
                except asyncio.TimeoutError as e:
                    caught_exception = e
                    log_message = "Request timed out:"
                    error = ErrorResponseInfo(error_msg="Request timed out", error_type="TimeoutError")
                except Exception as e:
                    caught_exception = e
                    log_message = "Unexpected error during request processing:"
                    error = ErrorResponseInfo(error_msg=str(e), error_type=type(e).__name__)

                # Retry only a connection fault that produced no response at all. Once
                # `response` is set the server answered, so whatever failed afterwards
                # (a broken body, a bad status) is a real result and is reported as one.
                if (
                    caught_exception is not None
                    and response is None
                    and attempt < max_attempts - 1
                    and is_retryable_transport_error(caught_exception)
                ):
                    retries_attempted += 1
                    backoff = self.client.request_retry_backoff_sec * (2**attempt)
                    # Jitter keeps concurrent requests from resynchronizing into a burst
                    # against a server that is already shedding connections.
                    await sleep(backoff * (0.5 + random.random()))
                    logger.warning(
                        f"Retrying request after {type(caught_exception).__name__} (attempt {attempt + 2}/{max_attempts})."
                    )
                    continue

                # Deferred to here so a fault that is about to be retried does not log an
                # ERROR with a stack trace: only a fault we are giving up on is an error.
                if caught_exception is not None:
                    logger.error(log_message, exc_info=caught_exception)

                # Recovered means the request succeeded, not just that the transport fault
                # stopped recurring: a final attempt can return a failing status without
                # raising, and error is set on every such path.
                if retries_attempted and caught_exception is None and error is None:
                    retries_recovered = True
                break

            end_time = time.perf_counter()

            # Time this request lost to attempts that bought nothing. Computed once here
            # and handed to both the span below and the request metric further down, which
            # cannot derive it from a shared source: `info` is still None at this point on
            # the failure path, since process_failure runs after the span is recorded.
            #
            # A recovered request wasted everything before the attempt that answered. A
            # request that never answered has no answering attempt to stop the clock at, so
            # its whole life was waste. Gated on retries_attempted rather than on
            # `attempt_start != start`, which is always true -- the loop re-stamps the
            # attempt clock on the first attempt too.
            if retries_attempted:
                retry_wasted_sec = (attempt_start - start) if retries_recovered else (end_time - start)

            # Record OTEL metrics
            self._record_otel_metrics(
                span=span,
                data=data,
                response=response,
                info=info,
                response_content=response_content,
                error=error,
                start_time=start,
                end_time=end_time,
                retry_wasted_sec=retry_wasted_sec,
            )

        if caught_exception is not None and not info:
            info = await data.process_failure(
                response=response,
                config=self.client.api_config,
                tokenizer=self.client.tokenizer,
                exception=caught_exception,
                lora_adapter=lora_adapter,
            )

        if not info:
            info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=0)))
        # Set after the default-fill above: a request whose every attempt failed gets a
        # fresh InferenceInfo, which still needs to report the retries it burned.
        info.retries_attempted = retries_attempted
        info.retries_recovered = retries_recovered
        # The same value the span carries. start_time on the metric below stays at
        # dispatch, so the request's own latency still counts this time; the field says how
        # much of it was waste.
        info.retry_wasted_sec = retry_wasted_sec
        if data.labels:
            info.labels = data.labels
        if data.graph_event_id:
            info.graph_event_id = data.graph_event_id

        metric = RequestLifecycleMetric(
            stage_id=stage_id,
            session_id=data.session_id if isinstance(data.session_id, str) else None,
            request_data=request_data,
            response_data=response_content,
            info=info,
            error=error,
            start_time=start,
            end_time=end_time,
            scheduled_time=scheduled_time,
        )

        # Grab TTFT and TPOT thresholds from request headers if available for streaming requests with token-level timestamps
        if (
            metric.info
            and isinstance(metric.info.response_metrics, StreamedResponseMetrics)
            and metric.info.response_metrics.output_token_times
        ):
            ttft_threshold = None
            tpot_threshold = None
            slo_unit = getattr(self.client.api_config, "slo_unit", None) or "ms"

            default_ttft_header = f"x-slo-ttft-{slo_unit}"
            default_tpot_header = f"x-slo-tpot-{slo_unit}"
            ttft_header = getattr(self.client.api_config, "slo_ttft_header", None) or default_ttft_header
            tpot_header = getattr(self.client.api_config, "slo_tpot_header", None) or default_tpot_header

            combined_headers = {}
            if self.client.api_config.headers:
                for k, v in self.client.api_config.headers.items():
                    combined_headers[k.lower()] = v
            if data.headers:
                for k, v in data.headers.items():
                    combined_headers[k.lower()] = v

            if combined_headers:
                ttft_threshold = combined_headers.get(ttft_header.lower())
                tpot_threshold = combined_headers.get(tpot_header.lower())

                unit = slo_unit.lower()
                unit_to_s = {"s": 1.0, "ms": 0.001, "us": 0.000001}
                factor = unit_to_s.get(unit, 1.0)

                if ttft_threshold is not None and ttft_threshold != "default":
                    try:
                        metric.ttft_slo_sec = float(ttft_threshold) * factor
                    except ValueError:
                        logger.warning(f"Invalid TTFT SLO value: {ttft_threshold}")

                if tpot_threshold is not None and tpot_threshold != "default":
                    try:
                        metric.tpot_slo_sec = float(tpot_threshold) * factor
                    except ValueError:
                        logger.warning(f"Invalid TPOT SLO value: {tpot_threshold}")

        # Record the metric
        self.client.metrics_collector.record_metric(metric)

    async def close(self) -> None:
        await self.session.close()
