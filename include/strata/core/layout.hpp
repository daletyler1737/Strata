// include/strata/core/layout.hpp - the pack's LAYOUT, resolved by name and checked against the kernels.
//
// `WeightTable` answers "where are the bytes".  This answers "is this the tensor I think it is", and it is
// the layer where a pack and a kernel disagree - which is the failure this project keeps paying for.  Two
// examples, both real:
//
//   * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY heads.  The
//     cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so the term was 4x too big and
//     the error survived a round because it moved in the direction that TIGHTENS the budget.
//   * `docs/semantics.md` and the tensor manifest agree on every dimension here, but nothing CHECKED that a
//     named tensor had the shape the kernel reading it assumes.
//
// So this header does two things: it names the tensors per layer type, and it asserts every shape the
// kernels depend on.  A mismatch is reported with the tensor name, the shape found and the shape required,
// at LOAD time - not as a wrong number inside a GEMV at token 4000.
#pragma once

#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

/// The model's geometry, taken from `docs/semantics.md` and the artifact's own metadata.  Every field here
/// is a number a kernel depends on, so a change is a change to a kernel contract and not a tuning knob.
struct ModelGeometry {
    int64_t n_embd = 2560;
    int64_t n_layers = 48;
    int64_t qsa_interval = 4;      ///< every 4th layer is full attention: layers 3, 7, ... 47

    // GDN (36 layers)
    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 48;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 10240;   ///< 2*128*16 + 128*48
    int64_t ssm_value_dim = 6144;        ///< 128 * 48

    // QSA (12 layers)
    int64_t n_head = 24;
    int64_t n_head_kv = 2;
    int64_t head_dim = 256;
    int64_t idx_q_heads = 4;
    int64_t idx_key_dim = 128;

    // PLE n-gram order, 0 = no PLE head.  Strata's shipped Flash-Next pack uses 3; the two qwen3_5
    // targets have no ngram head at all and must not build the 28.8 GB table.
    int64_t ple_ngram_size = 3;

    // gated residual, on every layer.  A pack without `hc_*` tensors (qwen3_5_text, qwen3_5_moe_text)
    // carries the zeros here; every consumer must ask `has_hc()` before reading them.
    int64_t hc = 4;
    int64_t hc_lr = 320;

    // MoE, on every layer.  A DENSE pack has no `ffn_gate_inp.weight`; n_expert = 0 says so.
    int64_t n_expert = 512;
    int64_t n_ff = 640;

    // **A DENSE FFN'S HIDDEN EXPANSION, WHICH IS NOT `n_ff`.**  In a MoE pack `n_ff` is the PER-EXPERT width
    // (640) and a layer's k selected experts each use it.  A dense pack's three matrices are the whole hidden
    // expansion instead: Qwen3.8-27B's `intermediate_size` is 17408, 27x.  Reusing `n_ff` for both would size the
    // FFN 27x wrong - a finite, plausible, completely wrong answer, since the gate GEMV would read 17408 rows
    // out of a 640-row matrix and find the rest inside whatever follows it in the arena.
    // Zero means "no dense FFN" (a MoE pack); `ffn_dense` is only reachable when has_moe() is false.
    int64_t dense_ffn = 0;

    /// **THE WIDTH EVERY FFN BUFFER FOLLOWS: the gate/up/down expansion, and the shared expert's scratch.**
    ///
    /// Both the projection width and `shared_expert_scratch_bytes` are as wide as the FFN's hidden expansion, so
    /// they take the same number.  A MoE pack's is `n_ff` (640, per expert); a dense pack's is `dense_ffn`
    /// (17408).  Sizing a dense FFN's scratch with `n_ff` overruns it 27x - the gate GEMV writes n_ff floats into
    /// a region meant for 640 of them, straight through the up/down buffers.  Silent: no bounds check, no crash,
    /// just a corrupted neighbouring buffer.
    ///
    /// This is ONE function, here, because a second copy in layer.cpp (or in a test) is a second thing to
    /// disagree with - and a stale copy in a test asserts nothing at all.
    int64_t ffn_width() const { return has_moe() ? n_ff : dense_ffn; }

    /// WHICH SUBSYSTEMS THIS PACK HAS, derived from the numbers above instead of stored beside them.
    /// A second set of flags is a second thing to disagree with the geometry, which is what this struct
    /// exists to prevent - so each is a comparison against a field the pack already pins.
    bool has_moe() const { return n_expert > 0; }
    bool has_hc() const { return hc > 0; }
    bool has_indexer() const { return idx_q_heads > 0; }
    /// A pack whose config has no ngram head (the two qwen3_5 targets) must not build the table.
    bool has_ple() const { return ple_ngram_size > 0; }

    int64_t hc_dim() const { return hc * n_embd; }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    int64_t n_qsa_layers() const { return n_layers / qsa_interval; }
    int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
    /// Every expert the pack holds, `layer * n_expert + expert`.  The frequency profile and the expert
    /// cache index by this, so it must be the geometry's arithmetic and not a constant anywhere.
    int64_t n_experts_total() const { return n_layers * n_expert; }
};

/// True for the full-attention layers.  `docs/semantics.md` gives this twice over - `full_attention_interval
/// = 4` and an explicit `attention.compress_ratios` array - and this is the first of the two.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    return layer % g.qsa_interval == g.qsa_interval - 1;
}

/// One layer's tensors, resolved by NAME.  `get("attn_qkv.weight")` looks up `blk.<layer>.attn_qkv.weight`
/// and returns null if the pack does not have it - a null is information, because a GDN layer has no
/// `attn_q` and a QSA layer has no `attn_qkv`.
///
/// Holds a reference to the table; the table must outlive it.
class LayerView {
public:
    LayerView(const WeightTable& table, int64_t layer) : table_(&table), layer_(layer) {}

    int64_t layer() const { return layer_; }
    std::string name(const char* suffix) const;
    const WeightRef* get(const char* suffix) const { return table_->find(name(suffix)); }

private:
    const WeightTable* table_;
    int64_t layer_;
};

/// Every shape the kernels depend on, asserted for ONE layer.  Returns false and fills `err` with the first
/// mismatch, naming the tensor, what it has and what is required.
///
/// This is deliberately a separate function from `LayerView`: a caller that only wants the pointers should
/// not pay for the checks, and a caller that wants the checks should get ALL of them rather than the ones
/// its own call site happens to touch.
bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err);

/// `check_layer` over every layer, plus the cross-layer properties: exactly 12 QSA layers at the right
/// indices, every GDN layer having the GDN set and no QSA tensor, and vice versa.  Returns the first failure.
bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err);

}  // namespace strata::core
