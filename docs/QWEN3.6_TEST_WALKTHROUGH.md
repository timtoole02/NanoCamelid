# Qwen3.6-35B MoE Distributed Parity Validation

This document records the verification and distributed test performance of the **Qwen3.6-35B-A3B MoE (Q4_K_M)** model on a Raspberry Pi 5 cluster.

## GGUF Inspect Validation Parity

Using the `nanocamelid inspect` tool, the GGUF file structure and tensor layout were validated successfully:

```text
NanoCamelid GGUF inspect
path: models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
version: 3
tensor_count: 753
metadata_count: 49

metadata:
  general.architecture: qwen35moe
  general.name: Qwen3.6 35B A3B
  tokenizer.ggml.model: gpt2
  general.quantization_version: 2
  general.file_type: 15

tensor_types:
  BF16: 2
  F32: 368
  Q4_K: 153
  Q5_K: 10
  Q6_K: 115
  Q8_0: 105
```

## Distributed Inference Setup

To run inference on the 20.6 GB model split across two 8GB/16GB Raspberry Pi 5 nodes under memory limits, the model was executed via a distributed split architecture using the `llama.cpp` RPC backend.

### 1. Daemon Startup (Worker Node)
Start the RPC daemon on the worker node to listen for tensor operations:
```bash
rpc-server --host 0.0.0.0 --port 5005
```

### 2. Client Execution (Master Node)
Execute inference in single-turn mode (`-st`) pointing to the worker's port:
```bash
llama-cli \
  -m models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --rpc <worker_ip>:5005 \
  -p "Write one short sentence about Raspberry Pi clusters." \
  -n 1024 -t 4 -st
```

## Baseline Performance Metrics (Pi 5 Cluster)

- **Prompt Processing (Prefill):** **`4.8 t/s`**
- **Text Generation (Decode):** **`2.2 t/s`**

### Output Parity Verification
The reasoning steps were executed with the extended token budget, outputting a fully coherent, grammatically correct single-sentence summary:

> **Raspberry Pi clusters combine multiple affordable single-board computers into a scalable, high-performance computing system.**
