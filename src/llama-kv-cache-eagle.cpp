#include "llama-kv-cache-eagle.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

llama_kv_cache_eagle::llama_kv_cache_eagle(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse) :
    llama_kv_cache(model, type_k, type_v, v_trans, offload, unified, kv_size,
        n_seq_max, n_pad, n_swa, swa_type, filter, reuse) {

    uint32_t n_stream = unified ? 1 : n_seq_max;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*(1 + n_stream)*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    // [TAG_V_CACHE_VARIABLE]
    const uint32_t n_embd_k_gqa =            _hparams().n_embd_k_gqa(0);
    const uint32_t n_embd_v_gqa = !v_trans ? _hparams().n_embd_v_gqa(0) : _hparams().n_embd_v_gqa_max();

    const char * dev_name = "CPU";

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    if (offload) {
        auto * dev = model.dev_layer(0);
        buft = ggml_backend_dev_buffer_type(dev);

        dev_name = ggml_backend_dev_name(dev);
    }

    LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, -1, dev_name);

    ggml_context * ctx = ctx_for_buft(buft);
    if (!ctx) {
        throw std::runtime_error("failed to create ggml context for kv cache");
    }

    ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);

    ggml_format_name(k, "cache_k_l%d", -1);
    ggml_format_name(v, "cache_v_l%d", -1);

    std::vector<ggml_tensor *> k_stream;
    std::vector<ggml_tensor *> v_stream;

    for (uint32_t s = 0; s < n_stream; ++s) {
        k_stream.push_back(ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]));
        v_stream.push_back(ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]));
    }

    eagle_layer = kv_layer({ _hparams().n_layer, k, v, k_stream, v_stream, });

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (model.hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        _ctxs_bufs().emplace_back(std::move(ctx), buf);
    }
}

ggml_tensor * llama_kv_cache_eagle::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    // non-eagle
    if (il >= 0) return llama_kv_cache::get_k(ctx, il, n_kv, sinfo);
    // eagle
    auto * k = eagle_layer.k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == _hparams().n_embd_k_gqa(0));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            _hparams().n_embd_head_k, _hparams().n_head_kv(0), n_kv, ns,
            ggml_row_size(k->type, _hparams().n_embd_head_k),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache_eagle::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    // non-eagle
    if (il >= 0) return llama_kv_cache::get_v(ctx, il, n_kv, sinfo);
    // eagle
    auto * v = eagle_layer.v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= _hparams().n_embd_v_gqa(0));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!_v_trans()) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                _hparams().n_embd_head_v, _hparams().n_head_kv(0), n_kv, ns,
                ggml_row_size(v->type, _hparams().n_embd_head_v),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                      // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),              // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, _hparams().n_head_kv(0), _hparams().n_embd_head_v, ns,
            ggml_row_size(v->type, kv_size*_hparams().n_embd_head_v),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                           // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),              // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache_eagle::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    // non-eagle
    if (il >= 0) return llama_kv_cache::cpy_k(ctx, k_cur, k_idxs, il, sinfo);
    // eagle
    GGML_UNUSED(sinfo);

    ggml_tensor * k = eagle_layer.k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache_eagle::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    // non-eagle
    if (il >= 0) return llama_kv_cache::cpy_v(ctx, v_cur, v_idxs, il, sinfo);
    // eagle
    GGML_UNUSED(sinfo);

    ggml_tensor * v = eagle_layer.v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!_v_trans()) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}
