// src/core/layout.cpp - the shape checks.  See the header for why this exists.
#include "strata/core/layout.hpp"

#include <cstdio>

namespace strata::core {
namespace {

/// A required 2-D shape.  `ne0` is the CONTIGUOUS axis, matching the manifest and `s_gemv`'s convention
/// (`y[o] = sum_i x[i]*W[i][o]`, row o contiguous of length ne0).
///
/// A tensor is asked for when BOTH gates open, and they are two independent questions:
///   * `sub`  - does this PACK carry the subsystem at all?  A qwen3_5 pack has no hc, no indexer and no
///     PLE head, so asserting their tensors would make every qwen3_5 load fail on a tensor the
///     architecture does not have.  The assertion is right; the ARCHITECTURE is a different one.
///   * `layer` - is this LAYER of that kind?  The indexer lives on the QSA layers only, so it needs both
///     questions answered yes.
enum class WantSub { kAny, kHc, kIndexer, kMoe };
enum class WantLayer { kAny, kQsa, kGdn };

struct Want2 {
    const char* suffix;
    int64_t ne0;
    int64_t ne1;
    WantSub sub;
    WantLayer layer;
};

/// A required element count and ENGINE FORM for a 1-D tensor.
///
/// `kind` is not decoration.  The engine form of a 1-D tensor is whatever `WeightKind` the loader applied -
/// `F32` copies 4 B/elem, `Bf16InF32` re-rounds to 2 - and a consumer that casts either one to `const float*`
/// reads 2x its length.  **THAT IS NOT HYPOTHETICAL: `ffn_gate_inp_shexp.weight` is BF16, so the arena holds
/// 5120 B for 2560 elements, and `shared_expert` read it as f32.  Every one of the 2560 MoE outputs came out
/// non-finite** - a wrong answer loud enough to notice, which is the lucky version.  Reading 5120 B past the
/// end of a tensor inside a 4.5 GiB arena does not fault, so the same mistake on a tensor followed by
/// plausible bytes would have produced plausible logits.
struct Want1 {
    const char* suffix;
    int64_t elements;
    WeightKind kind;
    WantSub sub;
    WantLayer layer;
};

bool fail(std::string& err, const LayerView& v, const char* suffix, const char* what, int64_t got,
          int64_t want) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "layer %lld: %s %s is %lld, the kernels require %lld",
                  (long long) v.layer(), v.name(suffix).c_str(), what, (long long) got, (long long) want);
    err = buf;
    return false;
}

/// The two gates, as one question.  A qwen3_5 pack has no hc / indexer / PLE head and asserts none of their
/// tensors; the GDN and QSA sets are keyed on the layer type.
bool wanted(WantSub sub, WantLayer layer, const ModelGeometry& g, bool qsa) {
    const bool pack_ok = sub == WantSub::kAny     ? true
                       : sub == WantSub::kHc      ? g.has_hc()
                       : sub == WantSub::kIndexer ? g.has_indexer()
                       : sub == WantSub::kMoe     ? g.has_moe()
                                                 : false;  // ponytail: closed enum
    const bool layer_ok = layer == WantLayer::kAny ? true
                        : layer == WantLayer::kQsa ? qsa
                        : layer == WantLayer::kGdn ? !qsa
                                                  : false;  // ponytail: closed enum
    return pack_ok && layer_ok;
}

bool check_one(const WeightTable& t, const ModelGeometry& g, int64_t layer, std::string& err) {
    const LayerView v(t, layer);
    const bool qsa = is_qsa_layer(g, layer);

    // ---- the 2-D tensor set, per layer family.  `hc_*` and `indexer.*` are Flash-Next only: a qwen3_5
    // pack has neither and must not be asked for them.
    const Want2 want2[] = {
        // gated residual, EVERY layer - `gr_read` takes w_down (hc_lr, hc_dim) and w_up (hc_lr, hc_dim)
        // after its own transpose, so the pack's orientation is [hc_dim, hc_lr] and [hc_lr, hc_dim].
        {"hc_attn_down.weight", g.hc_dim(), g.hc_lr, WantSub::kHc, WantLayer::kAny},
        {"hc_attn_up.weight", g.hc_lr, g.hc_dim(), WantSub::kHc, WantLayer::kAny},
        {"hc_attn_inject.weight", g.hc_dim(), g.hc, WantSub::kHc, WantLayer::kAny},
        {"hc_ffn_down.weight", g.hc_dim(), g.hc_lr, WantSub::kHc, WantLayer::kAny},
        {"hc_ffn_up.weight", g.hc_lr, g.hc_dim(), WantSub::kHc, WantLayer::kAny},
        {"hc_ffn_inject.weight", g.hc_dim(), g.hc, WantSub::kHc, WantLayer::kAny},
        // MoE, EVERY layer.  A dense pack asserts only the shared-expert trio and skips the router.
        {"ffn_gate_inp.weight", g.n_embd, g.n_expert, WantSub::kMoe, WantLayer::kAny},
        {"ffn_gate_shexp.weight", g.n_embd, g.n_ff, WantSub::kAny, WantLayer::kAny},
        {"ffn_up_shexp.weight", g.n_embd, g.n_ff, WantSub::kAny, WantLayer::kAny},
        {"ffn_down_shexp.weight", g.n_ff, g.n_embd, WantSub::kAny, WantLayer::kAny},
        // GDN only
        {"attn_qkv.weight", g.n_embd, g.ssm_conv_channels, WantSub::kAny, WantLayer::kGdn},
        {"attn_gate.weight", g.n_embd, g.ssm_value_dim, WantSub::kAny, WantLayer::kGdn},
        {"ssm_out.weight", g.ssm_value_dim, g.n_embd, WantSub::kAny, WantLayer::kGdn},
        {"ssm_conv1d.weight", g.ssm_d_conv, g.ssm_conv_channels, WantSub::kAny, WantLayer::kGdn},
        {"ssm_alpha.weight", g.n_embd, g.ssm_v_heads, WantSub::kAny, WantLayer::kGdn},
        {"ssm_beta.weight", g.n_embd, g.ssm_v_heads, WantSub::kAny, WantLayer::kGdn},
        // QSA only
        {"attn_q.weight", g.n_embd, 2 * g.n_head * g.head_dim, WantSub::kAny, WantLayer::kQsa},
        {"attn_k.weight", g.n_embd, g.n_head_kv * g.head_dim, WantSub::kAny, WantLayer::kQsa},
        {"attn_v.weight", g.n_embd, g.n_head_kv * g.head_dim, WantSub::kAny, WantLayer::kQsa},
        {"attn_output.weight", g.n_head * g.head_dim, g.n_embd, WantSub::kAny, WantLayer::kQsa},
        // THE INDEXER.  `q_proj` is the QUERY count and `k_proj` is the KEY width, and they are different
        // numbers - conflating them is what made the planner's indexer term 4x too big in round 194.
        // qwen3_5 has full attention with no indexer, so these are asked of nobody.
        {"indexer.q_proj.weight", g.n_embd, g.idx_q_heads * g.idx_key_dim, WantSub::kIndexer, WantLayer::kQsa},
        {"indexer.k_proj.weight", g.n_embd, g.idx_key_dim, WantSub::kIndexer, WantLayer::kQsa},
    };
    for (const Want2& w : want2) {
        if (!wanted(w.sub, w.layer, g, qsa)) continue;
        const WeightRef* r = v.get(w.suffix);
        if (!r) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(w.suffix);
            return false;
        }
        if (r->ne0 != w.ne0) return fail(err, v, w.suffix, "ne0", r->ne0, w.ne0);
        if (r->ne1 != w.ne1) return fail(err, v, w.suffix, "ne1", r->ne1, w.ne1);
    }

    // ---- the 1-D set.  EVERY ONE OF THESE IS `F32` EXCEPT the shared expert's scalar gate, which is BF16 -
    // and that single exception is the one a reader would not guess, so it is written down rather than
    // inferred from the tensor count.
    const Want1 want1[] = {
        {"hc_attn_norm.weight", g.hc_dim(), WeightKind::F32, WantSub::kHc, WantLayer::kAny},
        {"hc_ffn_norm.weight", g.hc_dim(), WeightKind::F32, WantSub::kHc, WantLayer::kAny},
        {"ffn_gate_inp_shexp.weight", g.n_embd, WeightKind::Bf16InF32, WantSub::kAny, WantLayer::kAny},
        {"ssm_a", g.ssm_v_heads, WeightKind::F32, WantSub::kAny, WantLayer::kGdn},
        {"ssm_dt.bias", g.ssm_v_heads, WeightKind::F32, WantSub::kAny, WantLayer::kGdn},
        {"ssm_norm.weight", g.ssm_state_size, WeightKind::F32, WantSub::kAny, WantLayer::kGdn},
        {"attn_q_norm.weight", g.head_dim, WeightKind::F32, WantSub::kAny, WantLayer::kQsa},
        {"attn_k_norm.weight", g.head_dim, WeightKind::F32, WantSub::kAny, WantLayer::kQsa},
        {"indexer.q_norm.weight", g.idx_key_dim, WeightKind::F32, WantSub::kIndexer, WantLayer::kQsa},
        {"indexer.k_norm.weight", g.idx_key_dim, WeightKind::F32, WantSub::kIndexer, WantLayer::kQsa},
    };
    for (const Want1& w : want1) {
        if (!wanted(w.sub, w.layer, g, qsa)) continue;
        const WeightRef* r = v.get(w.suffix);
        if (!r) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(w.suffix);
            return false;
        }
        if (r->elements != w.elements) return fail(err, v, w.suffix, "elements", r->elements, w.elements);
        if (r->kind != w.kind) {
            // The byte count is what makes this a REAL check and not a label: a 4 B/elem tensor holds
            // `elements * 4`, a 2 B/elem one holds `elements * 2`, and a consumer reading it as the wrong one
            // walks off the end.
            const uint64_t want_bytes = (uint64_t) w.elements *
                                        (w.kind == WeightKind::Bf16InF32 ? 2u : 4u);
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "layer %lld: %s is engine form %d (%llu B), the kernels read it as form %d (%llu B)",
                          (long long) layer, v.name(w.suffix).c_str(), (int) r->kind,
                          (unsigned long long) r->bytes, (int) w.kind, (unsigned long long) want_bytes);
            err = buf;
            return false;
        }
    }
    return true;
}

}  // namespace

std::string LayerView::name(const char* suffix) const {
    return "blk." + std::to_string(layer_) + "." + suffix;
}

bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err) {
    if (layer < 0 || layer >= g.n_layers) {
        err = "layer " + std::to_string(layer) + " is outside 0.." + std::to_string(g.n_layers - 1);
        return false;
    }
    return check_one(table, g, layer, err);
}

bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err) {
    int64_t n_qsa = 0, n_gdn = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!check_one(table, g, l, err)) return false;
        if (is_qsa_layer(g, l)) {
            ++n_qsa;
        } else {
            ++n_gdn;
        }
    }
    // The split is derived twice over in `docs/semantics.md` - the interval and an explicit ratio array -
    // and both give 36 GDN and 12 QSA.  Counting them here is the check that the LAYER TYPE PREDICATE and
    // the pack agree, which no per-tensor shape check can see.
    if (n_qsa != g.n_qsa_layers() || n_gdn != g.n_gdn_layers()) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "layer split is %lld QSA / %lld GDN, the geometry says %lld / %lld",
                      (long long) n_qsa, (long long) n_gdn, (long long) g.n_qsa_layers(),
                      (long long) g.n_gdn_layers());
        err = buf;
        return false;
    }
    return true;
}

}  // namespace strata::core
