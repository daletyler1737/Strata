// expert_layout_load() against the REAL Qwen3.6-35B-A3B-IQ2_XXS pack that tools/iq_pack.py just wrote from the
// 9.94 GiB GGUF.  expert_layout_test builds its own two-shard Q2_0 world and cannot take a real pack, so this
// calls the loader the way layer.cpp does.
//
// Every expected number is read off the real pack, not chosen to pass:
//   41 layers (40 trunk + 1 MTP), 256 experts, n_embd 2048, n_ff 512,
//   8.60 GiB of experts served as 260 zero-copy tensors out of the original GGUF,
//   layer 40 quantized Q4_0 while layers 0-39 are IQ2_XS / IQ2_XXS.
//
// ponytail: this checks that the layout LOADS and that every span is inside the file.  It does not compute a
// token - that needs the CUDA build, since the native dot products live in ggml-cpu's kernels and the WSL side
// has no GPU.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: q35_pack_load_test <pack_dir>\n"); return 2; }
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    };
    if (!strata::kernels::cpu::native_experts_available()) {
        std::printf("built without STRATA_NATIVE_EXPERTS - nothing to check\n");
        return 77;                                                  // ctest's "skipped"
    }
    check(true, "native expert path present");

    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(argv[1], 41, 256, err)) {
        std::printf("expert_layout_load refused the real pack: %s\n", err.c_str());
        return 1;
    }
    check(true, "expert_layout_load accepted the real 9.94 GiB Qwen3.6 pack");

    // The geometry came out of the v5 header, not the compiled-in Flash-Next default: n_ff 512 against a
    // compiled-in 64 is a 8x difference and would size every activation wrong.
    strata::kernels::cpu::NativeFmt f;
    std::string e2;
    // 10 of the 41 layers are 16/17 - gate/up IQ2_XS, down IQ2_XS - and their blob is 843776 B, which is not the
    // 811008 B the other 30 layers use.  A blob-size check against either one number would refuse half the model.
    check(strata::kernels::cpu::native_fmt(16, 17, 2048, 512, f, e2), "native_fmt for a 16/17 layer");
    check(f.n_embd == 2048 && f.n_ff == 512, "NativeFmt carries the pack's 2048 x 512, not a default");
    check(f.bytes == 843776, "16/17 blob is 843776 B, the value native_experts.txt writes");
    check(strata::kernels::cpu::native_fmt(16, 16, 2048, 512, f, e2), "native_fmt for a 16/16 layer");
    check(f.bytes == 811008, "IQ2_XS/IQ2_XS blob is 811008 B");

    // Layer 40 is the MTP block and ships Q4_0, not IQ: the table is per layer, so this has to hold for it too.
    check(strata::kernels::cpu::native_fmt(8, 8, 2048, 512, f, e2), "native_fmt for layer 40's Q4_0/Q4_0");
    check(f.bytes == 3342336, "Q4_0 MTP blob is 3342336 B, the value native_experts.txt writes");

    // gu_off/up_off/down_off must partition the blob: that is what makes the SEPARATE ffn_gate_exps /
    // ffn_up_exps layout work, and this file uses the separate form rather than one fused tensor.
    check(f.up_off > 0 && f.down_off > f.up_off && f.down_off < f.bytes,
          "up_off and down_off sit inside the blob, ordered after each other");

    // generate.cpp walks `for (l = 0; l < lay.fmt.size(); ++l)` and refuses on gu_type < 0, so the loader must
    // size fmt() to what the table actually holds.  Qwen3.6 has 41 blocks and 41 rows, and the gate once saw a
    // "layer 41" - i.e. fmt was one longer than the table and the entry past the end was default-constructed.
    const auto& L = strata::kernels::cpu::expert_layout();
    check(L.fmt.size() == 41, "fmt() has exactly the table's 41 entries, no default-constructed 42nd");
    check(L.offset.size() == 41 && L.bytes.size() == 41, "offset()/bytes() are 41 long too");
    int unset = 0;
    for (size_t i = 0; i < L.fmt.size(); ++i)
        if (L.fmt[i].gu_type < 0 || L.fmt[i].d_type < 0) ++unset;
    check(unset == 0, "no entry is left unset by the table");

    std::printf("real Qwen3.6 pack: %s\n", fails ? "FAILED" : "all cases pass");
    return fails ? 1 : 0;
}
