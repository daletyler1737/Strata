// 阶段 3 自检：qwen35moe（Qwen3.6-35B-A3B）的架构门槛 + 几何读取。
//
// 纯 CPU，零 CUDA，不下载任何权重：现场写一个只有元数据的最小 GGUF，键名和真模型
// 一模一样（取自 llama-arch.cpp 的 "%s.<name>" 表），然后让 read_geometry 真跑一遍。
//
// 锁三件事：
//  1. 门槛接受 qwen35moe，并按它自己的前缀找键（不是硬编码 qwen4exp.）
//  2. 几何逐字段来自文件，不是留在 Flash-Next 的默认值
//  3. 路由专家宽度 = n_ff / expert_used_count，不是直接拿 shared 的宽度
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"

using strata::GgufFile;
using strata::MetaValue;

static int failures = 0;
static void expect(bool ok, const char* what, long long got = -1, long long want = -1) {
    if (!ok) { ++failures; std::printf("FAIL %s: got %lld, want %lld\n", what, got, want); }
}

// ---- 最小 GGUF 写入器：只写 KV，不写张量数据 ----
static void put_u32(std::vector<uint8_t>& v, uint32_t x) { for (int i = 0; i < 4; ++i) v.push_back((x >> (i * 8)) & 0xff); }
static void put_u64(std::vector<uint8_t>& v, uint64_t x) { for (int i = 0; i < 8; ++i) v.push_back((x >> (i * 8)) & 0xff); }
static void put_str(std::vector<uint8_t>& v, const char* s) {
    put_u64(v, std::strlen(s));
    for (const char* p = s; *p; ++p) v.push_back((uint8_t) *p);
}
// MetaType id，对�� gguf_reader.hpp 的 enum 逐个数：U8=0 I8 U16 I16 U32=4 I32 F32=6 BOOL STRING=8。
// 写死数字不如引用常量：
// enum 里挪一个位置，下面全部错位，而错位后表现为 "unexpected end of file"，跟真截断一模一样。
enum : uint32_t { kTypeU32 = 4, kTypeString = 8 };
static uint64_t g_nkv = 0;
static void kv_str(std::vector<uint8_t>& v, const char* k, const char* val) {
    put_str(v, k); put_u32(v, kTypeString); put_str(v, val); ++g_nkv;
}
static void kv_u32(std::vector<uint8_t>& v, const char* k, uint32_t val) {
    put_str(v, k); put_u32(v, kTypeU32); put_u32(v, val); ++g_nkv;
}

static std::vector<uint8_t> build_gguf(const char* arch, bool with_used_count, bool with_hyper,
                                       bool with_key_length = true) {
    std::string p = std::string(arch) + ".";
    std::vector<uint8_t> kv;
    g_nkv = 0;
    kv_str(kv, "general.architecture", arch);
    kv_u32(kv, (p + "embedding_length").c_str(), 2048);
    kv_u32(kv, (p + "block_count").c_str(), 40);
    kv_u32(kv, (p + "attention.head_count").c_str(), 16);
    kv_u32(kv, (p + "attention.head_count_kv").c_str(), 2);
    if (with_key_length) kv_u32(kv, (p + "attention.key_length").c_str(), 256);
    kv_u32(kv, (p + "full_attention_interval").c_str(), 4);
    kv_u32(kv, (p + "expert_count").c_str(), 256);
    if (with_used_count) kv_u32(kv, (p + "expert_used_count").c_str(), 8);
    kv_u32(kv, (p + "expert_shared_feed_forward_length").c_str(), 512);
    kv_u32(kv, (p + "ssm.state_size").c_str(), 128);
    kv_u32(kv, (p + "ssm.group_count").c_str(), 16);
    kv_u32(kv, (p + "ssm.time_step_rank").c_str(), 32);
    kv_u32(kv, (p + "ssm.conv_kernel").c_str(), 4);
    // Flash-Next 才有 hyper_connection —— 这里带上，read_geometry 必须无视它
    if (with_hyper) {
        kv_u32(kv, (p + "hyper_connection.count").c_str(), 4);
        kv_u32(kv, (p + "hyper_connection.low_rank").c_str(), 320);
    }
    std::vector<uint8_t> out;
    put_u32(out, 0x46554747u);  // "GGUF" little-endian
    put_u32(out, 3);            // version
    put_u64(out, 0);            // tensor count   (the reader reads tensors BEFORE the KV count)
    put_u64(out, g_nkv);                // kv count
    out.insert(out.end(), kv.begin(), kv.end());
    // 对齐到 data_start：reader 的默认对齐是 32（general.alignment 缺省值）
    while (out.size() % 32) out.push_back(0);
    return out;
}


// ponytail: GgufFile only opens by PATH, so the test writes its synthetic metadata-only file and reads it back.
// Adding an in-memory constructor to production code to suit a test is a bad trade.
static std::string write_tmp(const std::vector<uint8_t>& b, int n) {
    char name[256];
    std::snprintf(name, sizeof name, "geom_test_%d.gguf", n);
    std::ofstream f(name, std::ios::binary);
    f.write((const char*) b.data(), (std::streamsize) b.size());
    f.close();
    return std::string(name);
}

int main() {
    // ===== 1. 门槛 =====
    {
        auto b = build_gguf("qwen35moe", true, false);
        std::unique_ptr<strata::GgufFile> f(new strata::GgufFile(write_tmp(b, 1)));
        const std::string err = strata::check_architecture(*f);
        expect(err.empty(), "qwen35moe passes check_architecture", (long long) err.size(), 0);
        if (!err.empty()) std::printf("  arch err: %s\n", err.c_str());
    }
    {
        auto b = build_gguf("llama", true, false);
        std::unique_ptr<strata::GgufFile> f(new strata::GgufFile(write_tmp(b, 2)));
        const std::string err = strata::check_architecture(*f);
        expect(err.find("requires") != std::string::npos, "a foreign arch is refused", (long long) err.size(), 1);
    }
    {
        // 键前缀必须跟着 arch 走：qwen4exp. 的键放在 qwen35moe 文件里，读不到 -> 报错而不是静默通过
        auto b = build_gguf("qwen4exp", true, true);
        std::unique_ptr<strata::GgufFile> f(new strata::GgufFile(write_tmp(b, 3)));
        const std::string err = strata::check_architecture(*f);
        expect(err.empty(), "qwen4exp still passes (its own prefix)", (long long) err.size(), 0);
        if (!err.empty()) std::printf("  arch err: %s\n", err.c_str());
    }

    // ===== 2 & 3. 几何 =====
    {
        auto b = build_gguf("qwen35moe", true, true);
        std::unique_ptr<strata::GgufFile> f(new strata::GgufFile(write_tmp(b, 4)));
        strata::core::ModelGeometry g;      // 从 Flash-Next 默认值开始
        std::string err;
        expect(strata::read_geometry(*f, g, err), "read_geometry succeeds", 0, 0);
        if (!err.empty()) std::printf("  err: %s\n", err.c_str());

        // 逐字段：全部来自文件
        expect(g.n_embd == 2048, "n_embd", g.n_embd, 2048);
        expect(g.n_layers == 40, "n_layers", g.n_layers, 40);
        expect(g.n_head == 16, "n_head", g.n_head, 16);
        expect(g.n_head_kv == 2, "n_head_kv", g.n_head_kv, 2);
        expect(g.head_dim == 256, "head_dim", g.head_dim, 256);
        expect(g.qsa_interval == 4, "qsa_interval", g.qsa_interval, 4);
        expect(g.n_expert == 256, "n_expert", g.n_expert, 256);
        expect(g.has_moe(), "qwen35moe is MoE");

        // **THE WIDTH TRAP**: 文件里只有 shared 的 512。路由宽度必须是 512/8 = 64，
        // 直接拿 512 会让每个专家宽 8 倍。
        expect(g.n_ff == 64, "routed expert width = shared / expert_used_count", g.n_ff, 64);

        // SSM 派生宽度
        expect(g.ssm_state_size == 128, "ssm_state_size", g.ssm_state_size, 128);
        expect(g.ssm_k_heads == 16, "ssm_k_heads", g.ssm_k_heads, 16);
        expect(g.ssm_v_heads == 32, "ssm_v_heads", g.ssm_v_heads, 32);
        expect(g.ssm_value_dim == 4096, "ssm_value_dim = state*v_heads", g.ssm_value_dim, 4096);
        expect(g.ssm_conv_channels == 8192, "ssm_conv_channels = state*(2*k+v)", g.ssm_conv_channels, 8192);

        // 文件里明明带了 hyper_connection.count=4，qwen35moe 也不该有 hc
        expect(!g.has_hc(), "hyper_connection keys are ignored");
        expect(g.hc == 0, "hc is zeroed", g.hc, 0);
        expect(g.hc_lr == 0, "hc_lr is zeroed", g.hc_lr, 0);
        expect(!g.has_indexer(), "no indexer");
        expect(!g.has_ple(), "no PLE");
        expect(g.ffn_width() == 64, "MoE ffn_width is n_ff", g.ffn_width(), 64);

        // **THE SILENT ONE**: 默认值被逐字段换掉，而不是留下来
        const strata::core::ModelGeometry def;
        expect(g.n_embd != def.n_embd, "n_embd moved off the Flash-Next default");
        expect(g.n_layers != def.n_layers, "n_layers moved off the default");
        expect(g.n_head != def.n_head, "n_head moved off the default");
        expect(g.n_expert != def.n_expert, "n_expert moved off the default");
    }

    // ===== head_dim 缺失时按 llama.cpp 的规则推 =====
    {
        auto b = build_gguf("qwen35moe", true, false, /*with_key_length=*/false);
        std::unique_ptr<strata::GgufFile> f(new strata::GgufFile(write_tmp(b, 5)));
        strata::core::ModelGeometry g;
        std::string err;
        expect(strata::read_geometry(*f, g, err), "key_length is optional", 0, 0);
        expect(g.head_dim == 2048 / 16, "head_dim derived as n_embd/n_head", g.head_dim, 128);
    }

    // ===== 缺 block_count 必须报错，不能沿用默认 48 =====
    {
        auto b = build_gguf("qwen35moe", true, false);
        std::string needle = "qwen35moe.block_count";
        auto pos = std::search(b.begin(), b.end(), needle.begin(), needle.end());
        if (pos != b.end()) *pos = 'z';
        std::unique_ptr<strata::GgufFile> f(new strata::GgufFile(write_tmp(b, 6)));
        strata::core::ModelGeometry g;
        std::string err;
        expect(!strata::read_geometry(*f, g, err), "a missing block_count is refused");
        expect(err.find("block_count") != std::string::npos, "the error names the key");
    }

    if (failures == 0) std::printf("geometry-qwen35moe: all cases pass\n");
    else std::printf("\ngeometry-qwen35moe: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}