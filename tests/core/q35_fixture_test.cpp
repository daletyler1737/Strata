// read_geometry against a real GGUF written by gguf-python (not the hand-rolled byte writer the other
// geometry test uses).  Same rules, different writer: if the reader only agrees with our own writer's quirks
// this fails.  The file is built by tools/make_q35_fixture.py into STRATA_Q35_FIXTURE.
//   usage: q35_fixture_test <with-routed-ff.gguf> <without-routed-ff.gguf>
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/layout.hpp"

#include <cstdio>
#include <cstring>
#include <string>

using strata::core::ModelGeometry;

static int fails = 0;
static void want(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); ++fails; }
}

// A file carrying expert_feed_forward_length takes the routed width from it; one without it must derive
// shared / expert_used_count.  Both are checked because a file WITH the key never reaches the division,
// so testing only it cannot tell the division from its absence.
static void check_common(const std::string& path) {
    strata::GgufFile f(path);
    want(!f.metadata().empty(), "the fixture opens and carries metadata");

    // the arch gate must accept qwen35moe (it is white-listed)
    std::string err;
    want(strata::check_architecture(f).empty(), "qwen35moe passes check_architecture");

    ModelGeometry g;
    std::string ge;
    want(strata::read_geometry(f, g, ge), "read_geometry accepts the fixture");
    if (fails) std::fprintf(stderr, "  %s\n", ge.c_str());

    // every field must come from the file, not from ModelGeometry's Flash-Next defaults
    want(g.n_embd == 2048, "n_embd is the file's 2048");
    want(g.n_layers == 2, "n_layers is the file's 2");
    want(g.n_head == 16, "n_head is 16");
    want(g.n_head_kv == 2, "n_head_kv is 2");
    want(g.n_expert == 256, "n_expert is 256");
    want(g.has_moe(), "it is a MoE");
    // the fixture carries no attention.key_length, so head_dim must be derived, not left at the 256 default
    want(g.head_dim == 2048 / 16, "head_dim is derived n_embd/n_head");
    // 256 experts must not be mistaken for 512
    want(g.n_expert != 512, "n_expert is not read as Flash-Next's 512");

}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <routed-ff.gguf> <no-routed-ff.gguf>\n", argv[0]);
        return 2;
    }
    { // the file that states its routed width: 64, NOT the shared expert's 512
        check_common(argv[1]);
        strata::GgufFile f(argv[1]);
        ModelGeometry g; std::string e;
        want(strata::read_geometry(f, g, e), "routed-ff file reads");
        want(g.n_ff == 64, "routed-ff file: n_ff is the stated 64, not the shared 512");
        want(g.ffn_width() == 64, "routed-ff file: ffn_width is 64");
    }
    { // the file without it: 512 / 8 = 64, so the division must happen to give the same 64
        check_common(argv[2]);
        strata::GgufFile f(argv[2]);
        ModelGeometry g; std::string e;
        want(strata::read_geometry(f, g, e), "no-routed-ff file reads");
        want(g.n_ff == 512 / 8, "no-routed-ff file: n_ff is the shared 512 divided by top-8");
        want(g.ffn_width() == 512 / 8, "no-routed-ff file: ffn_width is the divided width");
    }
    if (fails == 0) std::printf("q35 fixture (gguf-python): all cases pass\n");
    return fails ? 1 : 0;
}
