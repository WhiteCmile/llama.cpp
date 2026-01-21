#include "models.h"

llm_build_qwen3eagle::llm_build_qwen3eagle(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    bool flag = n_tokens == 1;
    // input for eagle
    ggml_tensor * emb_inp;
    ggml_tensor * hid_inp_1 = nullptr;
    ggml_tensor * hid_inp_2 = nullptr;
    ggml_tensor * hid_inp_3 = nullptr;

    ggml_tensor * layer_mask = nullptr;
    ggml_tensor * reverse_mask = nullptr;
    ggml_tensor * ONE = nullptr;
    // ONLY FOR DECODE
    if (flag) {
        layer_mask = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_layer);
        ggml_set_input(layer_mask);
        ggml_set_name(layer_mask, "layer_mask");
        cb(layer_mask, "layer_mask", -1);

        ONE = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        ggml_set_input(ONE);
        ggml_set_name(ONE, "ONE");
        cb(ONE, "ONE", -1);
        reverse_mask = ggml_sub(ctx0, layer_mask, ONE);
        reverse_mask = ggml_neg_inplace(ctx0, reverse_mask);
        cb(reverse_mask, "reverse_mask", -1);
    }

    inpL = build_inp_embd(model.tok_embd);
    emb_inp = inpL;

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        bool flag_layer = (il>1 && il<34)?flag:false;

        ggml_tensor * layer_input = inpL;
        ggml_tensor * inpSA = inpL;

        // hard-code for n_layer=36
        if (il == 2) hid_inp_1 = inpL;
        if (il == 18) hid_inp_2 = inpL;
        if (il == 33) hid_inp_3 = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            // TODO: for convenience, we do not masked the Q generation phase
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

            if (flag_layer) {
                // We mask the unnecessary attention layer
                cur = build_layer_masked_attn(inp_attn, 
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, 
                        nullptr, nullptr, nullptr, 
                        layer_mask,
                        1.0f/sqrtf(float(n_embd_head)), il);
            }
            else {
                // We build the attention layer as usual
                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (flag_layer) {
            // We mask the unnecessary FFN layer
            cur = build_layer_masked_ffn(cur, 
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    layer_mask,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
        }
        else {
            // We build the FFN layer as usual
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

        // Adapter
        if (flag_layer) {
            layer_input = build_norm(layer_input, model.layers[il].adapter_norm, nullptr, LLM_NORM_RMS, il);
            cb(layer_input, "adapter_norm", il);
            layer_input = build_layer_masked_ffn(
                layer_input,
                model.layers[il].adapter_up, nullptr, nullptr,
                model.layers[il].adapter_gate, nullptr, nullptr,
                model.layers[il].adapter_down, nullptr, model.layers[il].adapter_scale,
                nullptr, reverse_mask, LLM_FFN_SILU, LLM_FFN_PAR, il
            );
            // layer_input = build_ffn(
            //     layer_input,
            //     model.layers[il].adapter_up, nullptr, nullptr,
            //     model.layers[il].adapter_gate, nullptr, nullptr,
            //     model.layers[il].adapter_down, nullptr, model.layers[il].adapter_scale,
            //     nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il
            // );
            cb(layer_input, "l_adapter", il);
            layer_input = ggml_add(ctx0, layer_input, inpSA);
            cb(layer_input, "adapter_output", il);
        }

        // move input for this layer to input for next layer
        if (flag_layer) {
            cur = build_layer_masked_bypassing(
                cur, layer_input, 
                layer_mask, il
            );
            cb(cur, "l_out_masked", il);
            ggml_build_forward_expand(gf, cur);
        }

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    ggml_tensor * eagle_cur = nullptr;

    // eagle cal
    {
        // prepare input
        emb_inp = build_norm(emb_inp,
                    model.eagle_input_norm, NULL,
                    LLM_NORM_RMS, -1);
        cb(emb_inp, "eagle_emb_inp", -1);
        hid_inp_1 = ggml_concat(ctx0, hid_inp_1, hid_inp_2, 0);
        hid_inp_1 = ggml_concat(ctx0, hid_inp_1, hid_inp_3, 0);
        cb(hid_inp_1, "eagle_hid_concat", -1);
        hid_inp_1 = build_lora_mm(model.eagle_fc, hid_inp_1);
        cb(hid_inp_1, "eagle_hid_fc", -1);
        eagle_cur = build_norm(hid_inp_1,
                        model.eagle_hidden_norm, NULL,
                        LLM_NORM_RMS, -1);
        cb(eagle_cur, "eagle_hid_inp", -1);
        eagle_cur = ggml_concat(ctx0, emb_inp, eagle_cur, 0);
        cb(eagle_cur, "eagle_inp", -1);

        // attn
        {
            inpL = hid_inp_1; // save for res
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.eagle_q_proj, eagle_cur);
            cb(Qcur, "Qcur", -1);

            ggml_tensor * Kcur = build_lora_mm(model.eagle_k_proj, eagle_cur);
            cb(Kcur, "Kcur", -1);

            ggml_tensor * Vcur = build_lora_mm(model.eagle_v_proj, eagle_cur);
            cb(Vcur, "Vcur", -1);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", -1);
            cb(Kcur, "Kcur", -1);
            cb(Vcur, "Vcur", -1);

            eagle_cur = build_attn(inp_attn,
                            model.eagle_o_proj, nullptr,
                            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), -1);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, eagle_cur, inpL);
        cb(ffn_inp, "ffn_inp", -1);

        // feed-forward network
        eagle_cur = build_norm(ffn_inp,
                        model.eagle_ffn_norm, NULL,
                        LLM_NORM_RMS, -1);
        cb(eagle_cur, "ffn_norm", -1);

        eagle_cur = build_ffn(eagle_cur,
                        model.eagle_up_proj,   NULL, NULL,
                        model.eagle_gate_proj, NULL, NULL,
                        model.eagle_down_proj, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, -1);
        cb(eagle_cur, "ffn_out", -1);

        eagle_cur = ggml_add(ctx0, eagle_cur, ffn_inp);
    }

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        emb_inp = ggml_get_rows(ctx0, emb_inp, inp_out_ids);
        eagle_cur = ggml_get_rows(ctx0, eagle_cur, inp_out_ids);
    }

    ggml_tensor * router_cur = nullptr;
    // router cal
    {
        router_cur = ggml_concat(ctx0, emb_inp, eagle_cur, 0);
        cb(router_cur, "router_input", -1);
        router_cur = build_norm(router_cur, model.router_norm, nullptr, LLM_NORM_RMS, -1);
        cb(router_cur, "router_norm", -1);
        router_cur = build_ffn(
            router_cur,
            model.router_up, nullptr, nullptr,
            model.router_gate, nullptr, nullptr,
            model.router_down, model.router_down_b, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, -1
        );
        cb(router_cur, "router_logits", -1);
        router_cur = ggml_sigmoid_inplace(ctx0, router_cur);
        router_cur = ggml_round_inplace(ctx0, router_cur);
        cb(router_cur, "router_mask", -1);

        res->router_mask = router_cur;
        ggml_build_forward_expand(gf, router_cur);
    }

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
