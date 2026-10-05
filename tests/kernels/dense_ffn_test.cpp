// 阶段 3 自检：稠密 FFN（Qwen3.8-27B）。
//
// 纯 CPU，零 CUDA。它调的是 layout.hpp 里那个真的 `ModelGeometry::ffn_width()`，
// 不是一份副本——否则变异源文件时测试照样通过，等于什么都没锁。
// 锁两件事：
//  1. `dense_ffn` 与 `n_ff` 必须分开 —— 复用会让 FFN 宽度错 27 倍。
//  2. `shared_expert_scratch_bytes` 的宽度跟着 FFN 走，不是跟着 n_ff 走。
//
// 第 2 条本该在 kernel 里测，但 scratch 越界是静默的（写穿相邻缓冲，不崩），
// 所以这里锁的是"传给 scratch 计算的那个宽度"这个决定本身。
#include <cstdint>
#include <cstdio>

#include "strata/core/layout.hpp"

using strata::core::ModelGeometry;

static int failures = 0;
static void expect(bool ok, const char* what, long long got = -1, long long want = -1) {
    if (!ok) { ++failures; std::printf("FAIL %s: got %lld, want %lld\n", what, got, want); }
}

int main() {
    // --- Qwen3.8-27B: 64 层, n_embd 5120, intermediate_size 17408, 稠密, 无 MoE ---
    ModelGeometry m27;
    m27.n_embd = 5120; m27.n_layers = 64; m27.qsa_interval = 4;
    m27.n_expert = 0;        // 稠密
    m27.n_ff = 0;            // 稠密 pack 没有 per-expert 宽度
    m27.dense_ffn = 17408;   // intermediate_size
    m27.hc = 0; m27.hc_lr = 0; m27.idx_key_dim = 0; m27.idx_q_heads = 0; m27.ple_ngram_size = 0;

    // --- Qwen3.6-35B-A3B: MoE, n_ff = 512, 256 experts ---
    ModelGeometry m35;
    m35.n_embd = 2048; m35.n_layers = 40; m35.qsa_interval = 4;
    m35.n_expert = 256; m35.n_ff = 512; m35.dense_ffn = 0;
    m35.hc = 0; m35.hc_lr = 0; m35.idx_key_dim = 0; m35.idx_q_heads = 0; m35.ple_ngram_size = 0;

    // --- Flash-Next: MoE, n_ff = 640 ---
    ModelGeometry fn;
    fn.n_embd = 2560; fn.n_layers = 48; fn.qsa_interval = 4;
    fn.n_expert = 512; fn.n_ff = 640; fn.dense_ffn = 0;
    fn.hc = 4; fn.hc_lr = 320; fn.idx_key_dim = 128; fn.ple_ngram_size = 3;

    // 1. has_moe() 分流
    expect(!m27.has_moe(), "Qwen3.8-27B is dense");
    expect(m35.has_moe(), "Qwen3.6-35B is MoE");
    expect(fn.has_moe(), "Flash-Next is MoE");

    // 2. **THE REGRESSION**: 稠密 FFN 的宽度不是 n_ff。
    //    n_ff 保持 0（稠密没有 per-expert 宽度），实际宽度走 dense_ffn。
    expect(m27.n_ff == 0, "dense pack has no per-expert n_ff", m27.n_ff, 0);
    expect(m27.ffn_width() == 17408, "Qwen3.8-27B FFN width is intermediate_size", m27.ffn_width(), 17408);
    expect(m27.ffn_width() != m27.n_ff, "FFN width is NOT n_ff (the 27x bug)");

    // 3. MoE pack 的宽度仍是 n_ff，且不受 dense_ffn=0 影响
    expect(m35.ffn_width() == 512, "Qwen3.6-35B FFN width is n_ff", m35.ffn_width(), 512);
    expect(fn.ffn_width() == 640, "Flash-Next FFN width is n_ff", fn.ffn_width(), 640);

    // 4. 稠密 scratch 会比 MoE scratch 大很多 —— 这就是越界的那 27 倍。
    //    scratch 宽 n_ff 的时候按 640 分配，稠密要 17408。
    expect(m27.ffn_width() / fn.ffn_width() == 27, "dense scratch is 27x Flash-Next's",
           m27.ffn_width() / fn.ffn_width(), 27);

    // 5. 一个 MoE pack 若被误设了 dense_ffn，宽度必须仍是 n_ff（MoE 优先）
    ModelGeometry both = m35;
    both.dense_ffn = 17408;
    expect(both.ffn_width() == 512, "MoE wins over a stray dense_ffn", both.ffn_width(), 512);

    // 6. 默认几何（Flash-Next）不受影响
    ModelGeometry d;
    expect(d.ffn_width() == 640, "default geometry FFN width", d.ffn_width(), 640);

    // 7. **THE STALE-n_ff CASE.**  A dense pack whose metadata still carries a leftover `n_ff` must still use
    //    `dense_ffn`.  This is not hypothetical: `has_moe()` keys off `n_expert`, which is 0 here while `n_ff` is
    //    whatever the reader left behind.  A guard that reads "if n_ff != 0 use n_ff" gets this wrong, and for
    //    every fixture above it is indistinguishable from the correct code — which is exactly why it survived.
    ModelGeometry stale = m27;
    stale.n_ff = 640;
    expect(!stale.has_moe(), "stale fixture is still dense");
    expect(stale.ffn_width() == 17408, "a leftover n_ff must not size a dense FFN",
           stale.ffn_width(), 17408);
    // ...and the leftover is still there, unchanged, for whatever else reads it.
    expect(stale.n_ff == 640, "the stale n_ff is left alone for other consumers", stale.n_ff, 640);

    if (failures == 0) std::printf("dense-ffn: all cases pass\n");
    else std::printf("\ndense-ffn: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
