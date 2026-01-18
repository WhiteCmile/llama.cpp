我在大模型中。进行动态跳层。在每次推理开始，我会利用一个预测器，预测哪些层是需要执行的。之后，我就可以在一开始拿到这些需要执行的层。我规定一开始的两层是必须要执行的。
我现在的场景是，GPU显存不够放所有层，因此需要把一些曾放在CPU上计算。而由于我一开始就知道哪些层是需要的，因此我可以动态异步搬运层。

我现在的想法是，将GPU分为静态和动态两部分。我会离线的统计哪些层是经常激活的，那些层是不经常激活的。之后，经常激活的层会一直放在Gpu上，而GPU上还会留一小部分空间进行动态加载层。即在一开始的两层计算开始，我就将后面的需要的但不在GPU上的层搬运过来。

但这里有一个问题：我要保证cudagraph能够复用，那么在GPU上动态的部分应该怎么处理.
我现在的想法是，建立一整个计算图，对于静态在gpu上的层，每层输入输出都是固定的，对于动态的空间，我只留一个槽位，这个槽位的输入输出固定，如果走过的这个层是不需要跳的，则为他随机生成权重填到这个槽位，然后我会通过该算子来将输入直接映射到输出，相当于跳层。对于需要执行的不在gpu上的层，我将权重异步搬运到槽位（即知道需要哪些层且pcie空闲时我就开始搬运），将上个层的输出拷贝到槽位的输入，将槽位的输出拷贝到下一层的输入位置，

跳层算子我已经写好了，不需要再管，我现在在写搬运逻辑。
我现在流程是这样的。我的预测器和大模型要建到一个图里。因此图内部不能涉及到cpu的指令，因此我要在图建立之前新开一个线程。然后收到到slot空闲的信号就往里搬运权重。在图内部，有一个event，是确认这个slot权重到位信号才进行计算。

思路如下：

核心机制：


## 1. 全局同步变量
参考：
```cpp
// 全局（或封装到 model 中）
std::mutex copy_mutex;
std::condition_variable copy_cv;
bool copy_requested = false;
bool copy_done = false;
cudaEvent_t copy_complete_event;

// 初始化（程序启动时）
cudaEventCreate(&copy_complete_event);
```

## 2. 修改 llm_build_qwen3，插入 Host Callback
假设layer_mask是预测器的输出，我要在得到layermask之后一开始插入callback，通知外部线程：“现在可以搬权重了”
参考代码
```cpp
// 插入 host callback：触发权重搬运
    ggml_tensor* trigger = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(trigger, "prefetch_trigger");

    auto host_callback = [](ggml_tensor* t) {
        // 此函数在 graph 执行时被调用（host 线程）
        {
            std::lock_guard<std::mutex> lock(copy_mutex);
            copy_requested = true;
            copy_done = false;
        }
        copy_cv.notify_one(); // 唤醒搬运线程
    };

```

## 3. 外部线程搬运
对于不在gpu上的但需要执行的层，（结合layer_mask），我要按照model.get_slot_index_for_layer(il, 1, static_gpu_layers);映射关系搬运到对应的slot，这个线程是异步的。但是对于每个slot，需要创立一个信号，这个slot的权重使用过了，才可以开始往这个slot搬运下一个权重。

## 4.后续节点
Graph 中后续节点 等待搬运完成（通过 CUDA Event）再进行slot计算


## 5. 具体实现细节

### 5.1 修改 llama_slot 结构体
在 `llama-model.h` 中，为 `llama_slot` 添加 CUDA Event 用于同步：
```cpp
struct llama_slot {
    // ... 现有字段 ...
    
    // dynamic loading
    cudaEvent_t weight_ready_event = nullptr;
};
```

### 5.2 初始化和销毁 Event
在 `llama_model::create_slots_idv` 中为每个 slot 创建 Event：

在 `llama_model` 析构函数中销毁 Event：
```cpp
for (auto & slot : slots) {
    if (slot.weight_ready_event) {
        cudaEventDestroy(slot.weight_ready_event);
    }
}
```

### 5.3 添加搬运线程到 llama_context
在 `llama-context.h` 中添加成员：
```cpp
// dynamic loading
std::thread weight_transfer_thread;
std::atomic<bool> stop_thread{false};
std::atomic<bool> transfer_requested{false};
std::mutex transfer_mutex;
std::condition_variable transfer_cv;
std::unordered_set<int> slots_to_transfer;
std::mutex transfer_set_mutex;
std::vector<int> layer_for_slot;
std::vector<bool> slot_ready;
cudaStream_t transfer_stream = nullptr;
```

在构造函数中初始化：
```cpp
slot_ready.assign(model.slots.size(), false);
layer_for_slot.assign(model.slots.size(), -1);
cudaStreamCreate(&transfer_stream);
start_transfer_thread();
```

在析构函数中清理：
```cpp
stop_transfer_thread();
if (transfer_stream) {
    cudaStreamDestroy(transfer_stream);
}
```

### 5.4 搬运线程实现
```cpp

void llama_context::weight_transfer_worker() {

    
    while (true) {
            std::unordered_set<int> to_remove; // 本次成功搬运的 slot
        
        {
            std::unique_lock<std::mutex> lock(transfer_mutex);
            transfer_cv.wait(lock, [this]() {
                return stop_thread.load() || transfer_requested.load();
            });
            
            if (stop_thread.load()) {
                break;
            }
            
            // 注意：这里不再清空 slots_to_transfer！
            // 而是在搬运成功后，再从集合中移除
        }

        // 在 transfer_set_mutex 保护下访问和修改 slots_to_transfer
        {
            LLAMA_LOG_INFO("权重搬运线程收到搬运请求，开始搬运权重");
            std::lock_guard<std::mutex> lock_set(transfer_set_mutex);
            
            // 遍历当前所有待搬运的 slot
            for (int slot : slots_to_transfer) {
                // 跳过未就绪的 slot（保留在集合中，下次再试）
                if (slot_ready[slot]) {
                    continue;
                }

                int il = layer_for_slot[slot];
                if (il == -1) {
                    to_remove.insert(slot); // 无效层？也移除避免死循环
                    continue;
                }
                
                auto copy_tensor = [&](ggml_tensor * src, ggml_tensor * dst) {
                    size_t size = ggml_nbytes(src);
                    cudaMemcpyAsync(ggml_get_data(dst), ggml_get_data(src), size,
                                    cudaMemcpyHostToDevice, transfer_stream);
                };
                
                copy_tensor(model.layers[il].ffn_norm, model.slots[slot].ffn_norm);
                copy_tensor(model.layers[il].ffn_up,   model.slots[slot].ffn_up);
                copy_tensor(model.layers[il].ffn_gate, model.slots[slot].ffn_gate);
                copy_tensor(model.layers[il].ffn_down, model.slots[slot].ffn_down);
                
                cudaEventRecord(model.slots[slot].weight_ready_event, transfer_stream);
                slot_ready[slot] = true; // 标记为已搬运
                
                to_remove.insert(slot); // 标记为已处理
            }

            // 从待搬运集合中移除已处理的 slot
            for (int slot : to_remove) {
                slots_to_transfer.erase(slot);
            }
        }

        // 如果本次没有搬运任何 slot（全未就绪），可考虑短暂等待
        // 但通常由主线程再次触发 transfer_requested，所以可不做处理
        
        // 重置 transfer_requested 仅当 slots_to_transfer 为空？
        // 更安全的做法：只要还有 pending slot，就保持 transfer_requested = true
        // 但这样可能频繁唤醒。折中：每次搬运后都设为 false，由主线程重新 set
        transfer_requested.store(false);
    }
    
    LLAMA_LOG_INFO("权重搬运线程退出");
}

// 等待slot权重到位
void llama_context::wait_until_slot_ready(int slot_idx, cudaStream_t main_stream) {
    if (slot_idx < 0 || slot_idx >= (int)slot_ready.size()) return;
    if (!slot_ready[slot_idx]) {
        cudaStreamWaitEvent(main_stream, model.slots[slot_idx].weight_ready_event, 0);
        slot_ready[slot_idx] = true;
    }
}

void llama_context::release_slot(int slot_idx) {
    slot_ready[slot_idx] = false;
}
```

### 5.5 修改 llm_graph_params
在 `llama-graph.h` 中添加：
```cpp
struct llm_graph_params {
    // ... 现有字段 ...
    llama_context * ctx = nullptr;
};
```

在 `llama_context::graph_params` 中设置：
```cpp
params.ctx = const_cast<llama_context*>(this);
```

### 5.6 修改 llm_build_qwen3
在 `qwen3.cpp` 中：
```cpp
// 添加host callback来触发权重搬运
ggml_tensor* trigger = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
ggml_set_name(trigger, "prefetch_trigger");
auto host_callback = [this, layer_mask, &params]() {
    if (!layer_mask) return;
    int32_t * data = (int32_t*)ggml_get_data(layer_mask);
    std::unordered_set<int> to_transfer;
    for (int il = 0; il < n_layer; ++il) {
        bool use_static = static_gpu_layers.count(il);
        if (!use_static && data[il] == 1) {
            int slot_idx = model.get_slot_index_for_layer(il, 1, static_gpu_layers);
            to_transfer.insert(slot_idx);
        }
    }
    {
        std::lock_guard<std::mutex> lock(params.ctx->transfer_set_mutex);
        params.ctx->slots_to_transfer = std::move(to_transfer);
    }
    params.ctx->transfer_requested.store(true);
    params.ctx->transfer_cv.notify_one();
};
ggml_set_host_callback(trigger, host_callback);
ggml_build_forward_expand(gf, trigger);

// 在循环中设置 layer_for_slot
for (int il = 0; il < n_layer; ++il) {
    bool use_static = static_gpu_layers.count(il);
    if (!use_static) {
        int slot_idx = model.get_slot_index_for_layer(il, 1, static_gpu_layers);
        params.ctx->layer_for_slot[slot_idx] = il;
    }
    // ... 其余代码 ...
    
    if (!use_static) {
        params.ctx->wait_until_slot_ready(slot_idx);
        // ... 使用 slot ...
    }
}
```
