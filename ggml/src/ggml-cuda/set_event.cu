#include "set_event.cuh"

void ggml_cuda_op_release_slot(
    ggml_backend_cuda_context &ctx,
    ggml_tensor * dst) {

    // 从 op_params 中取出 slot_idx
    int slot_idx = *(int*)(dst->op_params);  // 注意：op_params 是 char[16]，需强转

    cudaStream_t cuda_stream = (cudaStream_t)ctx.stream();  // 或直接使用 stream（看你的 backend 实现）

    cudaEvent_t * slot_free_events = (cudaEvent_t *)ctx.user_data;
    cudaEventRecord(slot_free_events[slot_idx], cuda_stream);
}