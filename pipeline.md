我在大模型中。进行动态跳层。在每次推理开始，我会利用一个预测器，预测哪些层是需要执行的。之后，我就可以在一开始拿到这些需要执行的层。我规定一开始的两层是必须要执行的。
我现在的场景是，GPU显存不够放所有层，因此需要把一些曾放在CPU上计算。而由于我一开始就知道哪些层是需要的，因此我可以动态异步搬运层。

我现在的想法是，将GPU分为静态和动态两部分。我会离线的统计哪些层是经常激活的，那些层是不经常激活的。之后，经常激活的层会一直放在Gpu上，而GPU上还会留一小部分空间进行动态加载层。即在一开始的两层计算开始，我就将后面的需要的但不在GPU上的层搬运过来。

但这里有一个问题：我要保证cudagraph能够复用，那么在GPU上动态的部分应该怎么处理.
我现在的想法是，建立一整个计算图，对于静态在gpu上的层，每层输入输出都是固定的，对于动态的空间，我只留一个槽位，这个槽位的输入输出固定，如果走过的这个层是不需要跳的，则为他随机生成权重填到这个槽位，然后我会通过该算子来将输入直接映射到输出，相当于跳层。对于需要执行的不在gpu上的层，我将权重异步搬运到槽位（即知道需要哪些层且pcie空闲时我就开始搬运），将上个层的输出拷贝到槽位的输入，将槽位的输出拷贝到下一层的输入位置，

跳层算子我已经写好了，不需要再管，我现在在写搬运逻辑。
我现在流程是这样的。我的预测器和大模型要建到一个图里。因此图内部不能涉及到cpu的指令，因此我要在图建立之前新开一个线程。然后收到到slot空闲的信号就往里搬运权重。在图内部，有一个event，是确认这个slot权重到位信号才进行计算。

思路如下：

核心机制：

* graph 中插入一个 Host Callback 节点
* 该 callback 通知外部线程：“现在可以搬权重了”
* 外部线程执行 cudaMemcpyAsync（到预分配的 slot）
* Graph 中后续节点 等待搬运完成（通过 CUDA Event）

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
