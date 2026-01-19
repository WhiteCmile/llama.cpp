#include "llama-impl.h"
#include "llama-context.h"
#include "models.h"

llm_build_qwen3::llm_build_qwen3(const llama_model & model, 
    const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    int n_slots = model.n_slots;

    ggml_tensor * layer_to_slot = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_layer);
    ggml_set_name(layer_to_slot, "layer_to_slot");
    cb(layer_to_slot, "layer_to_slot", -1);

    LLAMA_LOG_INFO("layer-slot mapping done");

    LLAMA_LOG_INFO("building graph");

    ggml_tensor * cur;
    ggml_tensor * inpL;

    bool is_decode = n_tokens == 1;
    ggml_tensor * layer_mask = NULL;

    //假设之前预测器已经输出 layer_mask
    if (is_decode) {
        auto n_layer_mask_elem = n_layer;
        layer_mask = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_layer_mask_elem);
        ggml_set_input(layer_mask);
        ggml_set_name(layer_mask, "layer_mask");
        cb(layer_mask, "layer_mask", -1);
    }

    // add node: when we got layer_mask, we start another trnasfer stream to asnync copy later executing but not on gpu layer weight to corresponding slots
    // The stream transfer layer weight in an layer order (1,2,3...). When slot_ready_event is recorded, the transfer stream will copy the layer weight to the slot.
    // When this transfer is done, it will record weight_ready_event for the layer.
    // And then next layer

    // one kernel is pnly responsible for one layer tensor copy

    int slot_idx= 0;
    ggml_cuda_layer_prefetch_ctx* layers_ctx = nullptr; 
    
    if (is_decode){
        // assign prefetch context for each layer
        layers_ctx = new ggml_cuda_layer_prefetch_ctx[n_layer];

        for (size_t il = 0; il < n_layer; ++il) {

            layers_ctx[il].layer_tensor.ffn_norm = model.layers[il].ffn_norm;
            layers_ctx[il].layer_tensor.ffn_up   = model.layers[il].ffn_up;
            layers_ctx[il].layer_tensor.ffn_gate = model.layers[il].ffn_gate;
            layers_ctx[il].layer_tensor.ffn_down = model.layers[il].ffn_down;

            layers_ctx[il].slot_tensor.ffn_norm = model.slots[slot_idx].ffn_norm;
            layers_ctx[il].slot_tensor.ffn_up   = model.slots[slot_idx].ffn_up;
            layers_ctx[il].slot_tensor.ffn_gate = model.slots[slot_idx].ffn_gate;
            layers_ctx[il].slot_tensor.ffn_down = model.slots[slot_idx].ffn_down;

            layers_ctx[il].slot_free_event     = model.slots[slot_idx].slot_free_event;
            layers_ctx[il].weight_ready_event  = model.slots[slot_idx].weight_ready_event;
        }

        ggml_tensor* prefetch_node = build_prefetch_weights(
            ctx0,
            layer_mask,           // tensor input
            n_layer,              // total layers
            layer_to_slot,         // layer to slot mapping tensor
            layers_ctx
        );
    ggml_set_name(prefetch_node, "prefetch_weights");
    ggml_build_forward_expand(gf, prefetch_node);
    }

    inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;
        ggml_tensor * layer_input = inpL;

        bool use_static = model.static_gpu_layers.count(il);
        

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (is_decode) {
                cur = build_layer_masked_attn(inp_attn, 
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, 
                        nullptr, nullptr, nullptr, 
                        layer_mask,
                        1.0f/sqrtf(float(n_embd_head)), il);
            } else {
                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        if (use_static) {

            // static layer: no change
            LLAMA_LOG_INFO("%s: layer %d using static FFN (GPU)\n", __func__, il);

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            if (is_decode) {
                cur = build_layer_masked_ffn(cur, 
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        layer_mask,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
            } else {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
            }
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            if (is_decode) {
                cur = build_layer_masked_bypassing(cur, layer_input, layer_mask, il);
                cb(cur, "l_out_masked", il);
                ggml_build_forward_expand(gf, cur);
            }
            inpL = cur;
        } else {
            // using dynamic slots
            LLAMA_LOG_INFO("%s: layer %d using dynamic FFN (GPU slot %d)\n", __func__, il, slot_idx);

            // TODO: add graph node to wait for the weight ready event. (weight_ready_event)
            // WAIT for weight_ready_event before using slot weights
            ggml_tensor* wait_node = ggml_cuda_wait_event(ctx0, layers_ctx[il].weight_ready_event);
            ggml_set_name(wait_node, ("wait_weight_" + std::to_string(il)).c_str());
            ggml_build_forward_expand(gf, wait_node);


            cur = build_norm(ffn_inp,
                    model.slots[slot_idx].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.slots[slot_idx].ffn_up,   NULL, NULL,
                    model.slots[slot_idx].ffn_gate, NULL, NULL,
                    model.slots[slot_idx].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            inpL = cur;

            // TODO: add graph node to release the slot. (slot_release_event)
            // RECORD slot_free_event after layer is done
            ggml_tensor* release_node = ggml_cuda_record_event(ctx0, layers_ctx[il].slot_free_event);
            ggml_set_name(release_node, ("release_slot_" + std::to_string(il)).c_str());
            ggml_build_forward_expand(gf, release_node);
        }
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}