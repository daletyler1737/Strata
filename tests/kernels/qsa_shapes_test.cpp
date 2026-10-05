// `qsa_shapes(ModelGeometry)`: a pack with no indexer must get DENSE selection out of the SAME kernels.
//
// The whole of phase 1 is one line - `if (!g.has_indexer()) s.idx_top_k = kTopkMaxCells` - so the test is one
// claim: for a pack with `idx_q_heads == 0`, `qsa_selection_width(n_kv, s) == n_kv` at every context, which is
// exactly the condition under which the existing top-512 path returns the whole cell range.  With that true,
// `qsa_layer` needs no new kernel and no new branch: it selects every cell, gathers them, and the same
// softmax-over-gathered-cells runs dense attention.  With it false, a no-indexer pack silently gets
// `idx_top_k = 2048` and would truncate a 10K context to 2048 cells - a wrong answer, not a fault.
//
// CPU-only: `qsa.hpp` includes `layout.hpp` and `weights.hpp`, neither of which mentions CUDA.
#include "strata/kernels/qsa.hpp"

#include <cstdio>
#include <cstdint>

using strata::core::ModelGeometry;
using strata::kernels::QsaShapes;
using strata::kernels::qsa_real_shapes;
using strata::kernels::qsa_shapes;
using strata::kernels::qsa_selection_width;

namespace {
int failures = 0;

void expect(bool ok, const char* what, const char* detail = "") {
    std::printf("  %-62s %s %s\n", what, ok ? "ok" : "FAIL", detail);
    if (!ok) ++failures;
}

/// The two real targets, straight out of bench/*.json.  Only the fields `qsa_shapes` reads are set; the rest
/// keep their defaults so a typo in this test cannot pass by accident.
ModelGeometry flash_next() {
    ModelGeometry g;
    g.n_head = 24; g.n_head_kv = 2; g.head_dim = 256;
    g.idx_q_heads = 4; g.idx_key_dim = 128;
    return g;
}
ModelGeometry qwen_35b() {   // qwen3_5_moe_text: 16 heads, head_dim 256, no indexer
    ModelGeometry g;
    g.n_head = 16; g.n_head_kv = 2; g.head_dim = 256;
    g.idx_q_heads = 0; g.idx_key_dim = 0;
    return g;
}
}  // namespace

int main() {
    std::printf("qsa_shapes_test\n");

    // ---- 1. Flash-Next keeps the shipped selection budget: the indexer is still what picks the cells.
    {
        const QsaShapes s = qsa_shapes(flash_next());
        expect(s.idx_top_k == 2048, "flash-next keeps idx_top_k = 2048 (sparse)", nullptr);
        expect(s.idx_n_head == 4 && s.idx_dim == 128, "flash-next keeps its indexer geometry", nullptr);
        expect(qsa_selection_width(100, s) == 100, "flash-next below the bound is dense", nullptr);
        expect(qsa_selection_width(3000, s) == 2048 + 4 - 1, "flash-next above the bound selects 2051", nullptr);
    }

    // ---- 2. A pack with no indexer selects EVERY cell, at every context.  This is the phase-1 contract.
    {
        const QsaShapes s = qsa_shapes(qwen_35b());
        char d[128];
        for (int64_t n : {int64_t(1), int64_t(2047), int64_t(2048), int64_t(4096), int64_t(32768)}) {
            std::snprintf(d, sizeof d, "qwen3.6-35b selects all %lld cells (no indexer)",
                          (long long) n);
            expect(qsa_selection_width(n, s) == n, d, nullptr);
        }
        // and the ceiling: `topk_512` refuses rather than truncates, so the budget must not sit BELOW it
        expect(s.idx_top_k >= strata::kernels::kTopkMaxCells,
               "qwen3.6-35b budget covers kTopkMaxCells", nullptr);
    }

    // ---- 3. The heads are copied from the geometry, not from `qsa_real_shapes`.  Flash-Next's defaults are
    //         24/2/256; if qwen3.6-35b's 16 heads leaked through, the scores and the GQA map would both be wrong
    //         and every one of them would still produce finite output.
    {
        const QsaShapes s = qsa_shapes(qwen_35b());
        expect(s.n_head == 16, "qwen3.6-35b carries its own n_head (16, not 24)", nullptr);
        expect(s.n_head_kv == 2 && s.head_dim == 256, "qwen3.6-35b keeps n_head_kv/head_dim", nullptr);
    }

    // ---- 4. The bug this replaces: with the OLD body (no `has_indexer()` branch) a no-indexer pack kept
    //         idx_top_k = 2048, which would silently truncate a 10K context to 2051 cells.  Assert the
    //         truncation directly so the regression is visible in the number, not only in a pass/fail.
    {
        const QsaShapes old_body = qsa_real_shapes();   // what the unconditional copy produced
        const int64_t truncated = qsa_selection_width(10000, old_body);
        std::printf("  (the old body would have selected %lld of 10000 cells)\n", (long long) truncated);
        expect(truncated < 10000, "old body truncates a 10K context - this is what we fixed", nullptr);
        expect(qsa_selection_width(10000, qsa_shapes(qwen_35b())) == 10000,
               "new body selects all 10000", nullptr);
    }

    std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "all cases pass", failures);
    return failures ? 1 : 0;
}
