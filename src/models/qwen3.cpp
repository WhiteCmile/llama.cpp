#include "llama-impl.h"
#include "models.h"

llm_build_qwen3::llm_build_qwen3(const llama_model & model, 
    const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    bool is_decode = n_tokens == 1;
    ggml_tensor * layer_mask = NULL;

    if (is_decode) {
        auto n_layer_mask_elem = n_layer;
        layer_mask = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_layer_mask_elem);
        ggml_set_input(layer_mask);
        ggml_set_name(layer_mask, "layer_mask");
        cb(layer_mask, "layer_mask", -1);
    }

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

    inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;
        ggml_tensor * layer_input = inpL;

        bool use_static = static_gpu_layers.count(il);

        // LLAMA_LOG_INFO("%s: layer %d: use_static=%s\n", __func__, il, use_static ? "true" : "false");

        if (!use_static) {
            int slot_idx = model.get_slot_index_for_layer(il, 1, static_gpu_layers);
            ctx->layer_for_slot[slot_idx] = il;
        }

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
            // ====== DEBUG: 静态层 ======
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
            // ====== dynamic slot ======
            int slot_idx = model.get_slot_index_for_layer(il, 1, static_gpu_layers);

            params.ctx->wait_until_slot_ready(slot_idx);

            // ====== DEBUG: 打印 slot 分配 ======
            LLAMA_LOG_INFO("%s: layer %d assigned to slot %d\n", __func__, il, slot_idx);
            GGML_ASSERT(slot_idx >= 0 && slot_idx < (int)model.slots.size());

            params.ctx->layer_for_slot[slot_idx] = il;

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