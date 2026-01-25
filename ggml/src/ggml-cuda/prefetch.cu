#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"
#include <vector>

struct llama_slot {
    
    // normalization
    struct ggml_tensor * ffn_norm         = nullptr;
  
    // ff
    struct ggml_tensor * ffn_gate     = nullptr; // w1
    struct ggml_tensor * ffn_down     = nullptr; // w2
    struct ggml_tensor * ffn_up       = nullptr; // w3

 
    // // tensor ready event for using data in the slot
    cudaEvent_t weight_ready_event = nullptr;

    // // slot free event for next cuda memcpy async
    cudaEvent_t slot_free_event = nullptr;
};

__global__ void set_layer_results_kernel(int32_t * dst_layer_to_slot, const int32_t * src_results, int n_layers) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_layers) {
        dst_layer_to_slot[i] = src_results[i];
    }
}




void ggml_cuda_op_wait_event(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    
    printf("ggml_cuda_op_wait_event");
    fflush(stdout);
    cudaEvent_t cur_weight_ready = *static_cast<cudaEvent_t*>(dst->extra); //the target event, extra info of the tensor
    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(ctx.device, 1), cur_weight_ready, 0));

    // // 获取事件地址（唯一标识）
    // void* ev_ptr = static_cast<void*>(cur_weight_ready);
    // // 获取流 ID（简化：用指针地址代表流）
    // void* stream_ptr = static_cast<void*>(ctx.stream());

    // printf("ggml_cuda_op_wait_event: "
    //        "tensor=%p, event=%p, stream=%p\n",
    //        static_cast<void*>(dst), ev_ptr, stream_ptr);
    // fflush(stdout);

}

void ggml_cuda_op_record_event(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaEvent_t cur_slot_free = *static_cast<cudaEvent_t*>(dst->extra); //the target event, extra info of the tensor
    CUDA_CHECK(cudaEventRecord(cur_slot_free, ctx.stream()));
}

void ggml_cuda_op_prefetch_weights(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {


    // GGML_LOG_INFO("%s: GGML Prefetching \n", __func__);

    auto* params = static_cast<ggml_cuda_prefetch_params*>(dst->extra);

    struct llama_slot * slots = (struct llama_slot *) params->slots_data;

    cudaStream_t transfer_stream = ctx.stream(ctx.device, 1);

    ggml_tensor* mask_tensor = dst->src[0];
    // ggml_tensor* slot_tensor = dst->src[1];
    auto * static_mask = params->static_mask;

    std::vector<int32_t> host_mask(params->n_layers);
    // std::vector<int32_t> host_slots(params->n_layers);

    // 5. 数据回传（Device to Host）
    CUDA_CHECK(cudaMemcpy(host_mask.data(), mask_tensor->data, params->n_layers * sizeof(int32_t), cudaMemcpyDeviceToHost));
    // printf("[PREFETCH] layer_mask copied to CPU. First few: %d %d %d...\n", host_mask[0], host_mask[1], host_mask[2]);

    // CUDA_CHECK(cudaMemcpy(host_slots.data(), slot_tensor->data, params->n_layers * sizeof(int32_t), cudaMemcpyDeviceToHost));
    // // printf("

    // const int32_t * layer_mask =  (const int32_t*) mask_tensor-> data;
    // int32_t * layer_to_slot =  (int32_t*) slot_tensor-> data;

    const int32_t * layer_mask = host_mask.data();
    int32_t layer_to_slot[params->n_layers] = {0};

    int32_t current_slot_idx = 0;
    int n_slots = 2;


    // -------------- schedule ---------------
    // 仿真时间计算变量
    float current_gpu_busy_until = 0.0f; 
    float current_transfer_finished_at = 0.0f;
    
    // 常量定义 (单位: ms)
    const float P = 15.0f;  
    const float TC = 10.0f;
    const float TG = 1.50;



    for (int il = 0; il < params->n_layers; ++il) {

        // GGML_LOG_INFO("%s: Processing layer %d\n", __func__, il);

        if (static_mask[il]) { //it
            current_gpu_busy_until += TG;
            continue; // skip static layers. This won't change
        }

        if (!layer_mask[il]) {
            continue; // mask=0: skip this layer
        }

        // 3. 核心决策逻辑：
        // 模拟搬运：如果决定搬运，搬运结束时间点
        float simulated_transfer_end = current_transfer_finished_at + P;
        // 模拟GPU执行：由于GPU必须等搬运完且GPU空闲，执行结束时间点
        float simulated_gpu_end = simulated_transfer_end + TG;
        
        //层开始计算的时刻：如果要开始计算还没搬运完，就不搬运。
        float simulated_cpu_begin = current_gpu_busy_until;

        if (simulated_gpu_end < simulated_cpu_begin){

            struct llama_slot & slot = slots[current_slot_idx];  //指定空闲slot

            auto& layer_ctx = params->layers_ctx[il];

            // printf("[PREFETCH] Layer %3d -> Slot %2d | EventWait: %p | EventRecord: %p\n", 
            //        il, 
            //        current_slot_idx, 
            //        layer_ctx.slot_free_event, 
            //        layer_ctx.weight_ready_event);
            // fflush(stdout); 

            cudaEvent_t cur_slot_free    = static_cast<cudaEvent_t>(slot.slot_free_event);
            cudaEvent_t cur_weight_ready = static_cast<cudaEvent_t>(slot.weight_ready_event);

            layer_ctx.slot_free_event = cur_slot_free; //make sure the laye event points to the slot event
            layer_ctx.weight_ready_event = cur_weight_ready;

            // printf("[PREFETCH] event got \n");

            // Wait for slot to be free
            cudaStreamWaitEvent(transfer_stream, cur_slot_free, 0);

            // transfer weights
            auto copy = [&](ggml_tensor* src, ggml_tensor* dst) {
                size_t sz = ggml_nbytes(src);


                cudaMemcpyAsync(dst->data, src->data, sz,
                            cudaMemcpyHostToDevice, transfer_stream);

                // GGML_LOG_INFO("[PREFETCH]  Async copy begin\n");
            };

            // copy(layer_ctx.layer_tensor.ffn_norm, layer_ctx.slot_tensor.ffn_norm);
            copy(layer_ctx.layer_tensor.ffn_up,   slot.ffn_up);
            copy(layer_ctx.layer_tensor.ffn_gate, slot.ffn_gate);
            copy(layer_ctx.layer_tensor.ffn_down, slot.ffn_down);

            // Signal ready
            cudaEventRecord(cur_weight_ready, transfer_stream);

            current_slot_idx += 1;

            if(current_slot_idx >= n_slots){
                current_slot_idx = 0; // recycle
            }

            current_transfer_finished_at = simulated_transfer_end;
            current_gpu_busy_until = simulated_gpu_end;

            layer_to_slot[il] = current_slot_idx; // signal it as tensor needing to find in slot and write in slot index
            // if 0：it,s on cpu

            // printf("[STRATEGY] Layer %d: MOVING to GPU. Est. Finish: %.2fms\n", il, simulated_gpu_end);
            // fflush(stdout);
        } else{
            current_gpu_busy_until = simulated_cpu_begin+TC;
            // printf("[STRATEGY] Layer %d: STAY on CPU. Est. Finish: %.2fms\n", il, current_gpu_busy_until);
            // fflush(stdout);
        }

    }

    cudaMemcpy(dst->src[1]->data, layer_to_slot, 
                       params->n_layers * sizeof(int32_t), 
                       cudaMemcpyHostToDevice);


}
