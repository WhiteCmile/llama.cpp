#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"

void ggml_cuda_op_release_slot(ggml_backend_cuda_context &ctx, ggml_tensor * dst);