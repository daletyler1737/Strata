// native_expert_supported() for Qwen3.6-35B-A3B's real per-layer type pairs.  The gate said "layer 0's experts
// are IQ2_XXS/IQ2_XS ... which this engine has no GPU kernels for" and exited, because 16 and 17 were in
// STRATA_GU_FMTS but not STRATA_D_FMTS - the dot kernels existed, only the down list omitted them.
//
// This runs on the host: native_expert_supported is a plain host predicate over the Fmt<> tables, so it needs
// no GPU to answer.  The pairs come from the real pack's native_experts.txt, all three blob sizes and all
// three type combinations present in one model.
#include "strata/kernels/iq_kernels.hpp"
#include <cstdio>

int main() {
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    };
    // gu 16/17 (IQ2_XXS/IQ2_XS), down 17 (IQ2_XS) - layers 0-4, 36-40 of the real pack.
    check(strata::kernels::native_expert_supported(16, 17, 2048, 512), "16/17 IQ2_XXS/IQ2_XS  (10 real layers)");
    check(strata::kernels::native_expert_supported(16, 16, 2048, 512), "16/16 IQ2_XXS/IQ2_XXS (30 real layers)");

    // The other down types that were already in STRATA_D_FMTS must be unaffected.  42 (IQ4_NL) is the pair that
    // exercises both roles at once; 20 (IQ4_NL's sibling, gu-only) and 7 (Q5_1) are NOT gate/up types, so
    // gu_qk() is 0 for them and they were refused before this change too - asking for them here would be testing
    // a pair no pack has ever shipped.
    check(strata::kernels::native_expert_supported(42, 42, 2048, 512), "42/42 IQ4_NL unchanged");
    check(strata::kernels::native_expert_supported(23, 23, 2048, 512), "23/23 unchanged");
    check(strata::kernels::native_expert_supported(8, 8, 2048, 512), "8/8 Q4_0 unchanged");
    check(!strata::kernels::native_expert_supported(20, 20, 2048, 512), "20/20 still refused (gu-only, as before)");

    // The dimension guard still bites: IQ2 needs 256 values per block, and a width that is not a multiple of
    // it cannot be split into blocks at all.  Adding types must not have removed that.
    check(!strata::kernels::native_expert_supported(16, 17, 2048, 100), "16/17 refused when n_ff % 256 != 0");
    check(!strata::kernels::native_expert_supported(16, 17, 130, 512), "16/17 refused when n_embd % 256 != 0");
    // A type with no down entry at all is still refused - 18 (IQ3_XXS) is gate/up only.
    check(!strata::kernels::native_expert_supported(18, 18, 2048, 512), "18/18 still refused (gate/up only)");

    std::printf("down-type coverage: %s\n", fails ? "FAILED" : "all cases pass");
    return fails ? 1 : 0;
}
