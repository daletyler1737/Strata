// tests/core/layout_subsystem_test.cpp - `check_all` against three architectures, not one.
//
// `check_layer` is the load-time gate: it names the first tensor whose shape disagrees with the kernels'
// contract.  Making a subsystem OPTIONAL changes what that gate asks for, and a gate that quietly stops
// asking is worse than no gate.  So this pins the behaviour with a hand-built WeightTable per
// architecture: Flash-Next (everything), Qwen3.6-35B-A3B (MoE, no hc/indexer/PLE), Qwen3.8-27B (dense, no
// MoE either).  No GPU, no pack, no network.
//
//   g++ -std=c++17 -Iinclude tests/core/layout_subsystem_test.cpp src/core/layout.cpp -o layout_test
//   ./layout_test
#include "strata/core/layout.hpp"

#include <cstdio>
#include <map>
#include <string>

using strata::core::ModelGeometry;
using strata::core::WeightKind;
using strata::core::WeightRef;
using strata::core::WeightTable;

namespace {

/// The smallest table that satisfies `check_all` for one architecture.  Built from the geometry's own
/// numbers, so a test that passes is evidence the ASSERTIONS agree with the geometry - not that a
/// hand-typed list happens to match.
WeightTable build_table(const ModelGeometry& g, bool with_hc, bool with_indexer, bool with_moe,
                        bool with_ple) {
    WeightTable t;
    (void)with_ple;  // PLE tensors are not in the checker's set; has_ple() gates the ngram table instead
    auto put2 = [&](const std::string& name, int64_t ne0, int64_t ne1) {
        WeightRef r;
        r.ne0 = ne0;
        r.ne1 = ne1;
        r.elements = ne0 * ne1;
        r.resident = true;
        t.insert(name, r);
    };
    auto put1 = [&](const std::string& name, int64_t elements, WeightKind kind) {
        WeightRef r;
        r.elements = elements;
        r.kind = kind;
        r.resident = true;
        t.insert(name, r);
    };

    for (int64_t l = 0; l < g.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        if (with_hc) {
            put2(p + "hc_attn_down.weight", g.hc_dim(), g.hc_lr);
            put2(p + "hc_attn_up.weight", g.hc_lr, g.hc_dim());
            put2(p + "hc_attn_inject.weight", g.hc_dim(), g.hc);
            put2(p + "hc_ffn_down.weight", g.hc_dim(), g.hc_lr);
            put2(p + "hc_ffn_up.weight", g.hc_lr, g.hc_dim());
            put2(p + "hc_ffn_inject.weight", g.hc_dim(), g.hc);
            put1(p + "hc_attn_norm.weight", g.hc_dim(), WeightKind::F32);
            put1(p + "hc_ffn_norm.weight", g.hc_dim(), WeightKind::F32);
        }
        if (with_moe) put2(p + "ffn_gate_inp.weight", g.n_embd, g.n_expert);
        put2(p + "ffn_gate_shexp.weight", g.n_embd, g.n_ff);
        put2(p + "ffn_up_shexp.weight", g.n_embd, g.n_ff);
        put2(p + "ffn_down_shexp.weight", g.n_ff, g.n_embd);
        put1(p + "ffn_gate_inp_shexp.weight", g.n_embd, WeightKind::Bf16InF32);

        if (strata::core::is_qsa_layer(g, l)) {
            put2(p + "attn_q.weight", g.n_embd, 2 * g.n_head * g.head_dim);
            put2(p + "attn_k.weight", g.n_embd, g.n_head_kv * g.head_dim);
            put2(p + "attn_v.weight", g.n_embd, g.n_head_kv * g.head_dim);
            put2(p + "attn_output.weight", g.n_head * g.head_dim, g.n_embd);
            put1(p + "attn_q_norm.weight", g.head_dim, WeightKind::F32);
            put1(p + "attn_k_norm.weight", g.head_dim, WeightKind::F32);
            if (with_indexer) {
                put2(p + "indexer.q_proj.weight", g.n_embd, g.idx_q_heads * g.idx_key_dim);
                put2(p + "indexer.k_proj.weight", g.n_embd, g.idx_key_dim);
                put1(p + "indexer.q_norm.weight", g.idx_key_dim, WeightKind::F32);
                put1(p + "indexer.k_norm.weight", g.idx_key_dim, WeightKind::F32);
            }
        } else {
            put2(p + "attn_qkv.weight", g.n_embd, g.ssm_conv_channels);
            put2(p + "attn_gate.weight", g.n_embd, g.ssm_value_dim);
            put2(p + "ssm_out.weight", g.ssm_value_dim, g.n_embd);
            put2(p + "ssm_conv1d.weight", g.ssm_d_conv, g.ssm_conv_channels);
            put2(p + "ssm_alpha.weight", g.n_embd, g.ssm_v_heads);
            put2(p + "ssm_beta.weight", g.n_embd, g.ssm_v_heads);
            put1(p + "ssm_a", g.ssm_v_heads, WeightKind::F32);
            put1(p + "ssm_dt.bias", g.ssm_v_heads, WeightKind::F32);
            put1(p + "ssm_norm.weight", g.ssm_state_size, WeightKind::F32);
        }
    }
    return t;
}

/// A flash-next-shaped geometry with the subsystems switched off, so one builder covers all three.
ModelGeometry flash_next() {
    ModelGeometry g;
    return g;
}

int failures = 0;

void expect(bool ok, const char* what, const std::string& detail = "") {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s: %s\n", what, detail.c_str());
    }
}

void case_all_subsystems() {
    const ModelGeometry g = flash_next();
    expect(g.has_moe() && g.has_hc() && g.has_indexer() && g.has_ple(),
           "flash-next has every subsystem");
    expect(g.n_experts_total() == 48 * 512, "flash-next expert count", std::to_string(g.n_experts_total()));
    const WeightTable t = build_table(g, true, true, true, true);
    std::string err;
    expect(strata::core::check_all(t, g, err), "flash-next passes check_all", err);
}

void case_qwen36_35b() {
    // Qwen3.6-35B-A3B: MoE 256 experts / top-8 / n_ff 512, 40 layers, no hc, no indexer, no PLE.
    ModelGeometry g;
    g.n_embd = 2048;
    g.n_layers = 40;
    g.ssm_v_heads = 32;
    g.ssm_value_dim = 128 * 32;
    g.ssm_conv_channels = 2 * 128 * 16 + 128 * 32;
    g.n_head = 16;
    g.n_expert = 256;
    g.n_ff = 512;
    g.hc = 0;
    g.hc_lr = 0;
    g.idx_q_heads = 0;
    g.idx_key_dim = 0;
    g.ple_ngram_size = 0;

    // Qwen3.6-35B-A3B keeps its MoE.
    expect(g.has_moe(), "qwen3.6-35b keeps its MoE");
    expect(!g.has_hc() && !g.has_indexer() && !g.has_ple(), "qwen3.6-35b drops hc/indexer/PLE");
    expect(g.n_experts_total() == 40 * 256, "qwen3.6-35b expert count", std::to_string(g.n_experts_total()));
    expect(g.n_qsa_layers() == 10 && g.n_gdn_layers() == 30, "qwen3.6-35b layer split",
           std::to_string(g.n_qsa_layers()) + "/" + std::to_string(g.n_gdn_layers()));

    const WeightTable t = build_table(g, false, false, true, false);
    std::string err;
    expect(strata::core::check_all(t, g, err), "qwen3.6-35b passes check_all", err);

    // The point of the case: a table WITHOUT the router must be refused, so the gate did not simply
    // stop asking.  has_moe() is true here, so a missing ffn_gate_inp is still a failure.
    const WeightTable no_router = build_table(g, false, false, false, false);
    expect(!strata::core::check_all(no_router, g, err), "qwen3.6-35b still demands its router", err);
}

void case_qwen38_27b() {
    // Qwen3.8-27B: DENSE.  No router, no hc, no indexer, no PLE.  64 layers, 16 full attention.
    ModelGeometry g;
    g.n_embd = 5120;
    g.n_layers = 64;
    g.n_head_kv = 4;
    g.n_expert = 0;
    g.n_ff = 17408;
    g.hc = 0;
    g.hc_lr = 0;
    g.idx_q_heads = 0;
    g.idx_key_dim = 0;
    g.ple_ngram_size = 0;

    expect(!g.has_moe() && !g.has_hc() && !g.has_indexer() && !g.has_ple(),
           "qwen3.8-27b is dense and bare");
    expect(g.n_experts_total() == 0, "dense pack has no experts", std::to_string(g.n_experts_total()));
    expect(g.n_qsa_layers() == 16 && g.n_gdn_layers() == 48, "qwen3.8-27b layer split",
           std::to_string(g.n_qsa_layers()) + "/" + std::to_string(g.n_gdn_layers()));

    const WeightTable t = build_table(g, false, false, false, false);
    std::string err;
    expect(strata::core::check_all(t, g, err), "qwen3.8-27b passes check_all", err);

    // A dense pack must NOT be able to smuggle a router past the gate.
    ModelGeometry moe_like = g;
    moe_like.n_expert = 8;
    expect(moe_like.has_moe(), "n_expert > 0 restores the MoE gate");
    expect(!strata::core::check_all(t, moe_like, err), "router demanded once n_expert > 0", err);
}

}  // namespace

int main() {
    case_all_subsystems();
    case_qwen36_35b();
    case_qwen38_27b();
    if (failures == 0) std::printf("layout_subsystem_test: all cases pass\n");
    return failures == 0 ? 0 : 1;
}