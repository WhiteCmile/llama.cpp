#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"



void ggml_cuda_op_wait_event(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaEvent_t ev = *static_cast<cudaEvent_t*>(dst->extra); //the target event, extra info of the tensor
    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), ev, 0));
}

void ggml_cuda_op_record_event(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaEvent_t ev = *static_cast<cudaEvent_t*>(dst->extra); //the target event, extra info of the tensor
    CUDA_CHECK(cudaEventRecord(ev, ctx.stream()));
}

void ggml_cuda_op_prefetch_weights(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // the dst tensor is just a placeholder, we use its extra to store the target gg
    auto* params = static_cast<ggml_cuda_prefetch_params*>(dst->extra);
    
    cudaStream_t transfer_stream = ctx.stream(ctx.device, 1); // stream_id=1

    // 获取 layer_mask 和 layer_to_slot（从 src）
    ggml_tensor* mask_tensor      = dst->src[0];      // shape [n_layers], I32
    ggml_tensor* layer_to_slot_tensor   = dst->src[1];      // shape [n_layers], I32

    const int32_t * layer_mask = (const int32_t *)mask_tensor->data;
    const int32_t * layer_to_slot = (const int32_t *)layer_to_slot_tensor->data;

    for (int il = 0; il < params->n_layers; ++il) {

        int slot_idx = layer_to_slot[il];
        if (slot_idx == -1) continue; // skip static layers. This won't change

        auto& layer_ctx = params->layers_ctx[il]; // layer_ctx按层索引

        const bool not_skip = layer_mask[il];

        if (!not_skip) {
            continue; // skip this layer
        }

        cudaEvent_t cur_slot_free    = static_cast<cudaEvent_t>(layer_ctx.slot_free_event);
        cudaEvent_t cur_weight_ready = static_cast<cudaEvent_t>(layer_ctx.weight_ready_event);

        // Wait for slot to be free
        cudaStreamWaitEvent(transfer_stream, cur_slot_free, 0);

        // transfer weights
        auto copy = [&](ggml_tensor* src, ggml_tensor* dst) {
            size_t sz = ggml_nbytes(src);
            cudaMemcpyAsync(dst->data, src->data, sz,
                           cudaMemcpyHostToDevice, transfer_stream);
        };

        copy(layer_ctx.layer_tensor.ffn_norm, layer_ctx.slot_tensor.ffn_norm);
        copy(layer_ctx.layer_tensor.ffn_up,   layer_ctx.slot_tensor.ffn_up);
        copy(layer_ctx.layer_tensor.ffn_gate, layer_ctx.slot_tensor.ffn_gate);
        copy(layer_ctx.layer_tensor.ffn_down, layer_ctx.slot_tensor.ffn_down);

        // Signal ready
        cudaEventRecord(cur_weight_ready, transfer_stream);
    }

}
