#include "convert.cuh"

#define CUDA_FILL_BLOCK_SIZE 256

// Layer-masked bypassing kernel
// If layer_mask[layer_id] == true: dst[i] = src0[i] (bypass/skip)
// If layer_mask[layer_id] == false: dst[i] = src1[i] (normal)
template <typename T>
static __global__ void layer_masked_bypassing_kernel(
    T * dst,
    const T * src0,
    const T * src1,
    const float * layer_mask,
    const int layer_id,
    const int64_t k
) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    
    // Read mask once (coalesced access)
    // const bool not_skip = layer_mask[layer_id] >= 0.5f;
    
    // Conditional assignment
    dst[i] = (layer_mask[layer_id] >= 0.5f) ? src0[i] : src1[i];
}

void ggml_cuda_op_layer_masked_bypassing(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * mask_tensor = dst->src[2];  // Layer mask tensor
    
    void * dst_d = dst->data;
    const void * src0_d = src0->data;
    const void * src1_d = src1->data;
    const float * layer_mask = (const float *)mask_tensor->data;
    const int layer_id = dst->layer_id;

    // {
    //     printf("We are processing bypassing for %s\n", dst->name);
    //     int32_t mask_value = -999;
    //     cudaMemcpy(&mask_value, (void *) (layer_mask + layer_id), sizeof(int32_t), cudaMemcpyDeviceToHost);

    //     printf("Layer %d mask value is %d\n", layer_id, mask_value);
    // }

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_are_same_shape(dst, src0));
    GGML_ASSERT(ggml_are_same_shape(dst, src1));

    const int64_t k = ggml_nelements(dst);
    const int64_t num_blocks = (k + CUDA_FILL_BLOCK_SIZE - 1) / CUDA_FILL_BLOCK_SIZE;

    // Dispatch based on data type
    switch (dst->type) {
        case GGML_TYPE_F32:
            layer_masked_bypassing_kernel<float><<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream>>>(
                (float *)dst_d,
                (const float *)src0_d,
                (const float *)src1_d,
                layer_mask,
                layer_id,
                k
            );
            break;
        case GGML_TYPE_F16:
            layer_masked_bypassing_kernel<half><<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream>>>(
                (half *)dst_d,
                (const half *)src0_d,
                (const half *)src1_d,
                layer_mask,
                layer_id,
                k
            );
            break;
        default:
            GGML_ABORT("unsupported type for layer_masked_bypassing");
    }
}