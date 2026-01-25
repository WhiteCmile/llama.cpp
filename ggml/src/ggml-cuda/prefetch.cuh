#include "common.cuh"

void ggml_cuda_op_wait_event(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_record_event(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_prefetch_weights(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

