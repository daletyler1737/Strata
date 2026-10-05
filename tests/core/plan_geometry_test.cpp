// `plan::Geometry` derived from `ModelGeometry`, and the indexer term of `kv_bytes_per_token`.
//
// The planner's whole job is to divide a fixed VRAM pool between KV and the expert cache, so a wrong
// `Geometry` returns a plan that cannot be honoured and the failure lands at token 4000 instead of startup.
// Both call sites passed the DEFAULT `Geometry{}` - Flash-Next's numbers - for every model, so a 27B whose
// indexer is zero-width was charged for one and whose expert count is 16x smaller got Flash-Next's slots.
//
// The one claim: a pack with no indexer pays ZERO for the indexer, and `geometry_of` reproduces Flash-Next's
// plan byte-for-byte from Flash-Next's `ModelGeometry` (so the derivation is not a silent regression).
//
// CPU-only: `plan.hpp` includes `layout.hpp` and `weights.hpp`, neither of which mentions CUDA.
#include "strata/plan/plan.hpp"

#include <cstdio>
#include <cstdint>

using strata::core::ModelGeometry;
using strata::plan::Geometry;
using strata::plan::geometry_of;
using strata::plan::kv_bytes_per_token;
using strata::plan::state_bytes;

namespace {
int failures = 0;

void expect(bool ok, const char* what, const char* = nullptr) {
    std::printf("  %-64s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

uint64_t mb(uint64_t b) { return b / (1024ull * 1024ull); }

/// Flash-Next exactly as `layout.hpp` ships it.
ModelGeometry flash_next() {
    ModelGeometry g;
    g.n_embd = 2560; g.n_layers = 48; g.qsa_interval = 4;
    g.n_head = 24; g.n_head_kv = 2; g.head_dim = 256;
    g.idx_q_heads = 4; g.idx_key_dim = 128;
    g.hc = 4; g.hc_lr = 320;
    g.n_expert = 512; g.n_ff = 640;
    g.ssm_state_size = 128; g.ssm_v_heads = 48; g.ssm_d_conv = 4; g.ssm_conv_channels = 10240;
    return g;
}
/// Qwen3.8-27B: dense, 64 layers, full attention every 4th, NO indexer.
ModelGeometry qwen_27b() {
    ModelGeometry g;
    g.n_embd = 5120; g.n_layers = 64; g.qsa_interval = 4;
    g.n_head = 24; g.n_head_kv = 4; g.head_dim = 256;
    g.idx_q_heads = 0; g.idx_key_dim = 0;
    g.hc = 0; g.hc_lr = 0;
    g.n_expert = 0; g.n_ff = 17408;
    g.ssm_state_size = 128; g.ssm_v_heads = 48; g.ssm_d_conv = 4;
    g.ssm_conv_channels = (4 - 1) * (128 * 16 + 2 * 16 * 128);
    return g;
}
}  // namespace

int main() {
    std::printf("plan_geometry_test\n");

    // ---- 1. Flash-Next is unchanged.  Its KV-per-token is `n_qsa(12) * (kv + idx)` with
    //         kv = 2*2*256 + (2*2*256/64)*2 = 1024 + 32 = 1056 B, and idx = 1*128/4 = 32 B
    //         -> 12 * 1088 = 13056 B/token.  If the derivation changes this, the shipped plan moves.
    {
        const Geometry p = geometry_of(flash_next());
        const uint64_t want = 12ull * (1056ull + 32ull);
        char d[160];
        std::snprintf(d, sizeof d, "flash-next: %lu B/token (was %lu)",
                      (unsigned long) kv_bytes_per_token(p), (unsigned long) want);
        expect(kv_bytes_per_token(p) == want, d);
        expect(p.n_layers == 48 && p.n_qsa_layers == 12 && p.n_kv_heads == 2 && p.head_dim == 256,
               "flash-next: layers/heads come from the geometry", nullptr);
        expect(p.indexer_key_dim == 128 && p.indexer_q_heads == 4, "flash-next: carries its indexer width");
    }

    // ---- 2. A pack with no indexer pays ZERO for it, not Flash-Next's 32 B/token/layer.
    {
        const Geometry p = geometry_of(qwen_27b());
        const uint64_t kv = 2ull * 4 * 256 + (2ull * 4 * 256 / 64) * 2;   // 2048 + 128 = 2176 B
        const uint64_t want = 16ull * kv;                                 // 16 of 64 layers
        char d[160];
        std::snprintf(d, sizeof d, "qwen3.8-27b: %lu B/token = KV only, no indexer term",
                      (unsigned long) kv_bytes_per_token(p));
        expect(kv_bytes_per_token(p) == want, d);
        expect(p.indexer_key_dim == 0, "qwen3.8-27b: zero indexer width");
        expect(p.indexer_key_dim == 0 && p.n_qsa_layers == 16 && p.n_kv_heads == 4,
               "qwen3.8-27b: 16 QSA layers, 4 KV heads, zero indexer width", nullptr);

        // THE REGRESSION, AS A NUMBER.  With `idx_layer` unguarded the arithmetic is
        //   `1 * 0 / 4 = 0` - which is ALREADY right, because the width is zero.  So the bug I am fixing is
        //   NOT that the indexer was charged; it is that `Geometry{}` was passed instead of the loaded
        //   geometry, so EVERY model was planned with Flash-Next's `indexer_key_dim = 128`.  The mutant that
        //   deletes the guard must therefore be caught by a pack that has Flash-Next's indexer WIDTH and no
        //   indexer, which is not a pack that exists - so the guard is defensive, not the fix.
        //
        // What the guard actually buys: a pack whose indexer is off but whose METADATA still carries a width
        // (the shape `plan::Geometry` describes - `indexer_key_heads`/`_key_dim` are independent fields).  A
        // zero-width pack is already correct without it.
        const uint64_t charged_old = 16ull * (kv + 32ull);
        std::printf("  (if a pack carried Flash-Next's 128-wide indexer width: %lu B/token more, %lu MB over 32K)\n",
                    (unsigned long) (charged_old - want),
                    (unsigned long) ((charged_old - want) * 32768ull / (1024ull * 1024ull)));

        // The charge IS proportional to the width - no predicate, no flag.  Zero width -> zero bytes.
        Geometry wider = p;
        wider.indexer_key_dim = 128;
        expect(kv_bytes_per_token(wider) == 16ull * (kv + 32ull),
               "a pack whose metadata carries a 128-wide indexer is charged 32 B/layer", nullptr);
        expect(kv_bytes_per_token(p) != kv_bytes_per_token(wider),
               "the width is the only thing separating the two charges", nullptr);
    }

    // ---- 3. The bug this replaces is the DEFAULT, not a wrong argument: both callers passed `Geometry{}`.
    //         So the loaded geometry must be what reaches the planner, and a qwen3_5 plan must differ from a
    //         Flash-Next plan.  Assert the plans differ rather than that some argument is passed - the
    //         argument is what got dropped in the first place.
    {
        const uint64_t fn = kv_bytes_per_token(geometry_of(flash_next()));
        const uint64_t qw = kv_bytes_per_token(geometry_of(qwen_27b()));
        expect(fn != qw, "the two packs' KV budgets differ (so Geometry{} was not silently right)");
        char d[160];
        std::snprintf(d, sizeof d, "over 32K: flash-next %lu MB vs qwen3.8-27b %lu MB",
                      (unsigned long) mb(fn * 32768ull), (unsigned long) mb(qw * 32768ull));
        expect(true, d);

        // `n_layers` is a field `geometry_of` must carry, and `n_gdn_layers()` is derived from it.  Two
        // models whose KV-per-token agrees must still differ here if their layer counts do.
        const Geometry a = geometry_of(flash_next());
        const Geometry b = geometry_of(qwen_27b());
        expect(a.n_layers == 48 && b.n_layers == 64, "geometry_of carries n_layers through", nullptr);
        expect(a.n_gdn_layers() == 36 && b.n_gdn_layers() == 48, "and n_gdn_layers derives from it", nullptr);
    }

    // ---- 4. `state_bytes` reads the GDN recurrence off the same geometry.  A qwen3_8-27b layer is
    //         128*48*128*4 + 3*conv_channels*4; with conv_channels from the config that is not Flash-Next's.
    {
        const uint64_t fn = state_bytes(geometry_of(flash_next()));
        const uint64_t qw = state_bytes(geometry_of(qwen_27b()));
        char d[160];
        std::snprintf(d, sizeof d, "recurrent state: flash-next %lu MB vs qwen3.8-27b %lu MB",
                      (unsigned long) mb(fn), (unsigned long) mb(qw));
        expect(fn != qw, d);
    }

    std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "all cases pass", failures);
    return failures ? 1 : 0;
}
