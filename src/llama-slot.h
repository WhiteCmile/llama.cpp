// #include "ggml-backend.h"
// #include "llama-model.h"
// #include <cstddef>
// #include <unordered_set>
// struct llama_slot {
//     ggml_backend_buffer_t buffer = nullptr;   // GPU/CUDA buffer
//     size_t                size   = 0;
//     ggml_backend_event_t  event  = nullptr;   // 权重就绪事件
//     bool                  initialized = false;
//     std::vector<llama_layer> layers;

//     bool init(ggml_backend_t backend, size_t max_weight_size);
//     // bool ensure_loaded(int layer) const;
//     void destroy();
//     bool copy_weight_async(const void * cpu_data, size_t data_size, ggml_backend_t backend);
//     int  get_slot_index_for_layer(int il, int num_slots, std::unordered_set<int> static_gpu_layers) const;
// };