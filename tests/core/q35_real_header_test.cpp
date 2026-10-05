// read_geometry() against the REAL bartowski Qwen3.6-35B-A3B-IQ2_XXS header.  The file does not have to be
// downloaded: GGUF's header is at the front, so the keys, the tensor names and their shapes are all readable
// from the first few MB.  What this pins is the metadata contract - the numbers below were read off the real
// file, not chosen to make a test pass:
//   arch qwen35moe, n_embd 2048, 41 blocks (40 trunk + 1 MTP), 16 heads / 2 KV heads, head_dim 256,
//   256 experts, top-8, expert_feed_forward_length 512 (the ROUTED width, same as the shared expert),
//   full_attention_interval 4, and SEPARATE ffn_gate_exps / ffn_up_exps (not the fused tensor).
#include "strata/artifact/gguf_reader.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    };
    strata::GgufFile g(argv[1]);
    check(strata::check_architecture(g).empty(), "architecture accepted (qwen35moe)");
    strata::core::ModelGeometry m;
    std::string err;
    if (!strata::read_geometry(g, m, err)) {
        std::printf("read_geometry refused the real file: %s\n", err.c_str());
        return 1;
    }
    check(m.n_embd == 2048, "n_embd 2048");
    check(m.n_layers == 41, "n_layers 41 (40 trunk + 1 MTP)");
    check(m.n_expert == 256, "n_expert 256");
    check(m.n_ff == 512, "n_ff 512 - the routed width, taken from expert_feed_forward_length");
    check(m.n_head == 16 && m.n_head_kv == 2, "16 heads, 2 KV heads");
    check(m.head_dim == 256, "head_dim 256 from attention.key_length");
    check(m.qsa_interval == 4, "full_attention_interval 4 -> 10 full + 31 GDN");
    check(m.has_moe() && !m.has_hc() && !m.has_indexer() && !m.has_ple(),
          "subsystems: MoE only (no hyper-connections, no indexer, no PLE)");
    check(m.n_qsa_layers() == 10 && m.n_gdn_layers() == 31, "10 full-attention layers, 31 GDN layers");
    check(m.ffn_width() == 512, "ffn_width() is the routed 512, not a MoE constant");
    std::printf("real qwen35moe header: %s\n", fails ? "FAILED" : "all cases pass");
    return fails ? 1 : 0;
}
