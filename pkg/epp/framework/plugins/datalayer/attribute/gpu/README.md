# GPU Attributes

This package defines the data structures for GPU hardware metrics collected from NVIDIA DCGM Exporter.

## `GPUUtilization`

A normalized GPU compute utilization value in [0.0, 1.0], derived from `DCGM_FI_DEV_GPU_UTIL` (which reports 0-100). For multi-GPU pods the extractor aggregates across visible devices using `max`.

- **Key**: `GPUUtilizationDataKey` (`GPUUtilization/dcgm-extractor`)

## Producers

The following plugins produce this attribute:

- **`dcgm-extractor`** (Data Layer): Extracts GPU utilization from the DCGM Exporter Prometheus endpoint.
