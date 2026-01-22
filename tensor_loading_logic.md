# Llama Model Tensor Loading Logic

## Overview
The `load_tensors` function in `llama-model.cpp` is responsible for loading model weights from disk and allocating them to appropriate backend buffers (CPU or GPU). This process involves creating tensor metadata, selecting backend buffer types based on layer types and configurations, and allocating memory buffers.

## Key Components

### 1. Buffer Type Lists
- **CPU Buffer List**: Created using `make_cpu_buft_list()` - includes CPU, ACCEL, and extra buffer types
- **GPU Buffer List**: Created using `make_gpu_buft_list()` for each GPU device - primarily GPU buffer types with CPU as fallback

### 2. Layer Assignment
Layers are assigned to devices based on:
- **Input Layer**: Always CPU (`pimpl->dev_input`)
- **Repeating Layers**: Distributed across devices using `get_layer_buft_list()` based on memory splits
- **Output Layer**: Assigned using `get_layer_buft_list(n_layer)`

### 3. Tensor Creation Process (`create_tensor` lambda)

#### Tensor Layer Types
- **LLM_TENSOR_LAYER_INPUT**: Token embeddings, always on CPU
- **LLM_TENSOR_LAYER_OUTPUT**: Output weights, assigned to output device
- **LLM_TENSOR_LAYER_REPEATING**: Layer-specific tensors, assigned to layer device

#### Buffer Selection Logic
1. Select `buft_list` based on layer type
2. Apply custom overrides (FFN/adapter CPU forcing)
3. Check tensor buffer type overrides
4. Use `select_weight_buft()` to choose compatible buffer type

#### Special Handling
- **FFN Tensors**: Can be forced to CPU via `ffn_layer_config`
- **Adapter Tensors**: Forced to CPU for layers 2-33 in QWEN3EAGLE
- **Duplicated Tensors**: Reuse existing tensors when possible

### 4. Context and Buffer Creation

#### Context Management
- One GGML context per buffer type (`ctx_map`)
- Contexts created on-demand via `ctx_for_buft()`

#### Buffer Allocation
- **MMAP Mode**: Use `ggml_backend_dev_buffer_from_host_ptr()` for memory-mapped files
- **Non-MMAP Mode**: Use `ggml_backend_alloc_ctx_tensors_from_buft()`
- Buffers locked in memory if `use_mlock` is enabled

### 5. Architecture-Specific Tensor Creation

#### Common Pattern
```cpp
for (int i = 0; i < n_layer; ++i) {
    // Create attention tensors
    layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {...}, 0);
    // Create FFN tensors
    layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {...}, 0);
    // Architecture-specific tensors
}
```

#### Special Cases
- **QWEN3EAGLE**: Includes adapter layers (2-33), Eagle components, and router
- **Input/Output Tensors**: Created once, not per-layer
- **Duplicated Tensors**: Output may reuse token embeddings

### 6. Memory Management
- **Buffer Usage**: Set to `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` for scheduling optimization
- **Memory Locking**: Applied to host buffers when requested
- **Memory Reporting**: Prints buffer sizes per type

### 7. Custom Modifications
- **FFN CPU Forcing**: Based on `ffn_layer_config` tensor
- **Adapter CPU Forcing**: Hardcoded for layers 2-33 in QWEN3EAGLE
- **Debug Logging**: Added to track tensor buffer assignments

## Layer-Specific Processing

### Token Embeddings (Input Layer)
- Always assigned to CPU buffer list
- Single tensor: `tok_embd`

### Output Layer (LM Head)
- Assigned to output device buffer list
- Includes: `output_norm`, `output`, `cls_out` (optional)

### Repeating Layers
- Distributed across devices based on memory splits
- Each layer contains:
  - Attention: `wq`, `wk`, `wv`, `wo`, norms
  - FFN: `ffn_gate`, `ffn_up`, `ffn_down`, norms
  - Architecture-specific: adapters (QWEN3EAGLE), MoE experts, etc.

## Buffer Assignment Flow

1. **Determine Layer Device**: Based on layer index and GPU layers setting
2. **Select Buffer List**: CPU/GPU based on device assignment
3. **Apply Overrides**: Custom logic for FFN/adapter CPU forcing
4. **Choose Buffer Type**: `select_weight_buft()` finds compatible type
5. **Create Context**: Get or create GGML context for buffer type
6. **Create Tensor**: Allocate tensor in appropriate context
7. **Allocate Buffer**: Create backend buffer for context tensors

## Debugging and Monitoring
- Added logging for tensor buffer assignments
- Memory usage reporting per buffer type
- Layer device assignment logging