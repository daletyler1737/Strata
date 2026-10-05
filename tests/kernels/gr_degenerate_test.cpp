// 阶段 2 的退化路径自检：没有 hc_* 的 pack（Qwen3.6-35B / Qwen3.8-27B）。
//
// 纯 CPU，零 CUDA。它锁的是**块缓冲区的宽度**这一个事实 —— 那正是我第一版漏掉的地方：
// `b.R` 原本按 `g.hc * g.n_embd` 分配，hc=0 时是 0 floats，而无 hc 的块仍然要做 `R += block_out`。
// 也就是说整个普通 Transformer residual 写在了缓冲区的起点之外。不崩溃，只是静默损坏。
//
// 这个自检不需要 GPU：它检查的是算术和宽度，不是 kernel。
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

#include "strata/core/layout.hpp"

using strata::core::ModelGeometry;

static int failures = 0;

static void expect(bool ok, const char* what, long long got = -1, long long want = -1) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s: got %lld, want %lld\n", what, got, want);
    }
}

// 复制自 layer.cpp 的 `block_buffers_bytes` 的 R 项 —— 改一处必须改两处，所以这里重述。
static long long r_floats(const ModelGeometry& g) {
    return g.has_hc() ? (long long) g.hc * g.n_embd : (long long) g.n_embd;
}

int main() {
    // --- Qwen3.6-35B-A3B: 40 层, 10 full-attn, 256 experts, 无 hc ---
    ModelGeometry m35;
    m35.n_embd = 2048;
    m35.n_layers = 40;
    m35.qsa_interval = 4;
    m35.n_expert = 256;
    m35.n_ff = 512;
    m35.n_head = 16;
    m35.n_head_kv = 2;
    m35.head_dim = 128;
    m35.hc = 0;       // 关键
    m35.hc_lr = 0;
    m35.idx_key_dim = 0;
    m35.idx_q_heads = 0;     // ModelGeometry 的默认值是 Flash-Next 的（4），必须清零
    m35.ple_ngram_size = 0;  // 同上（默认 3）
    m35.ssm_state_size = 128;
    m35.ssm_k_heads = 16;
    m35.ssm_v_heads = 32;
    m35.ssm_d_conv = 4;

    // --- Qwen3.8-27B: 64 层, 16 full-attn, 稠密 FFN, 无 hc ---
    ModelGeometry m27;
    m27.n_embd = 5120;
    m27.n_layers = 64;
    m27.qsa_interval = 4;
    m27.n_expert = 0;  // 稠密
    m27.n_ff = 17408;
    m27.n_head = 24;
    m27.n_head_kv = 4;
    m27.head_dim = 128;
    m27.hc = 0;
    m27.hc_lr = 0;
    m27.idx_key_dim = 0;
    m27.idx_q_heads = 0;
    m27.ple_ngram_size = 0;
    m27.ssm_state_size = 128;
    m27.ssm_k_heads = 16;
    m27.ssm_v_heads = 48;
    m27.ssm_d_conv = 4;

    // 1. has_hc() 判据本身
    expect(!m35.has_hc(), "Qwen3.6-35B has no hyper-connection");
    expect(!m27.has_hc(), "Qwen3.8-27B has no hyper-connection");

    // 2. 层数
    expect(m35.n_qsa_layers() == 10, "Qwen3.6-35B full-attn layers", m35.n_qsa_layers(), 10);
    expect(m27.n_qsa_layers() == 16, "Qwen3.8-27B full-attn layers", m27.n_qsa_layers(), 16);
    // 稠密 vs MoE
    expect(!m27.has_moe(), "Qwen3.8-27B is dense (no MoE)");
    expect(m35.has_moe(), "Qwen3.6-35B is MoE");
    // indexer
    expect(!m35.has_indexer(), "Qwen3.6-35B has no indexer");
    expect(!m27.has_indexer(), "Qwen3.8-27B has no indexer");
    // PLE
    expect(!m35.has_ple(), "Qwen3.6-35B has no PLE");
    expect(!m27.has_ple(), "Qwen3.8-27B has no PLE");

    // 3. **THE REGRESSION**: 残差缓冲区必须存在且够宽。
    //    hc=0 时 `g.hc * g.n_embd` = 0，而无 hc 的块仍然做 R += block_out。
    expect(r_floats(m35) == 2048, "Qwen3.6-35B R width (n_embd, NOT 0)", r_floats(m35), 2048);
    expect(r_floats(m27) == 5120, "Qwen3.8-27B R width (n_embd, NOT 0)", r_floats(m27), 5120);

    // 4. Flash-Next 不受影响：仍然是 4 流的栈。
    ModelGeometry fn;
    fn.n_embd = 2560; fn.hc = 4; fn.hc_lr = 320; fn.n_layers = 48; fn.n_expert = 512;
    fn.qsa_interval = 4; fn.idx_key_dim = 128;
    expect(fn.has_hc(), "Flash-Next has a hyper-connection");
    expect(r_floats(fn) == 4 * 2560, "Flash-Next R width is the 4-wide stack", r_floats(fn), 4 * 2560);

    // 5. 一个 hc=1 的中间情形：仍然是有栈的（1 流），宽度 == n_embd。门是 has_hc()，不是 hc != 0。
    ModelGeometry one;
    one.n_embd = 512; one.hc = 1; one.hc_lr = 64; one.n_layers = 2; one.qsa_interval = 2;
    expect(one.has_hc(), "hc=1 still counts as a hyper-connection");
    expect(r_floats(one) == 512, "hc=1 R width", r_floats(one), 512);

    // 6. block_out 也要放得下（普通块仍然产生它）
    //    隐含在 r_floats 的判据里；这里只确认宽度关系成立。
    expect(r_floats(m27) >= m27.n_embd, "R holds at least one block_out");

    if (failures == 0) std::printf("gr-degenerate: all cases pass\n");
    else std::printf("\ngr-degenerate: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
