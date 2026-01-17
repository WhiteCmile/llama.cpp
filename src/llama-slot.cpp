// // llama_slot.cpp
// #include "llama-slot.h"
// #include <unordered_map>
// #include <unordered_set>

// bool llama_slot::init(ggml_backend_t backend, size_t max_weight_size_per_layer) {
//     const int num_slots = 4;
//     size_t total_size = num_slots * max_weight_size_per_layer;

//     // 对齐到 256 字节（ggml 要求）
//     total_size = (total_size + 255) & ~255ULL;

//     buffer = ggml_backend_alloc_buffer(backend, total_size);
//     if (!buffer) {
//         fprintf(stderr, "%s: failed to allocate slot buffer of size %zu\n", __func__, total_size);
//         return false;
//     }

//     size = total_size;
//     // event = ggml_backend_event_new(backend); // 用于同步权重加载完成

//     // 预分配 layers 容器（仅用于索引，实际 data 指向 buffer）
//     layers.resize(num_slots);

//     // 注意：这里不初始化 tensor，只预留空间
//     // 实际 tensor 的 data 指针将在 copy_weight_async 时设置

//     initialized = true;
//     return true;
// }

// void llama_slot::destroy() {
//     if (buffer) {
//         ggml_backend_buffer_free(buffer);
//         buffer = nullptr;
//     }
//     if (event) {
//         ggml_backend_event_free(event);
//         event = nullptr;
//     }
//     initialized = false;
// }

// bool llama_slot::copy_weight_async(const void * cpu_data, size_t data_size, ggml_backend_t backend) {
//     if (!initialized || data_size > size) return false;

//     // 3. 异步拷贝 CPU → GPU
//     void * gpu_ptr = ggml_backend_buffer_get_base(buffer);
//     // 注意：ggml_backend_tensor_set_async 需要 tensor，但我们只有 raw ptr
//     // 所以我们临时构造一个 dummy tensor
//     static struct ggml_init_params params = { .mem_size = 1024, .no_alloc = true };
//     static struct ggml_context * ctx = ggml_init(params);

//     struct ggml_tensor * dummy = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, data_size / sizeof(float));
//     dummy->data = gpu_ptr;
//     dummy->buffer = buffer;

//     // 异步写入
//     ggml_backend_tensor_set_async(backend, dummy, cpu_data, 0, data_size);

//     // 4. 记录事件（如果支持）
//     if (event) {
//         ggml_backend_event_record(event, backend);
//     }

//     return true;
// }

// int llama_slot::get_slot_index_for_layer(int il, int num_slots, std::unordered_set<int> static_gpu_layers) const{
//     // 只对动态层计算索引
//     if (static_gpu_layers.count(il)) return -1;  // 静态层不用 slot

//     // 统计动态层序号（从 0 开始）
//     int dynamic_idx = 0;
//     for (int i = 0; i < il; ++i) {
//         if (!static_gpu_layers.count(i)) dynamic_idx++;
//     }

//     return dynamic_idx % num_slots;  // 轮询分配到 slot
// }

// void llama_slot_free(llama_slot * slot) {
//     delete slot;
// }