// include/strata/artifact/gguf_reader.hpp - generated from src/artifact/gguf_reader.cpp by
// scripts/split_artifact.py.  Header-only on purpose: the reader is one translation unit's worth of
// code with no state to hide, and a header-only split cannot introduce a duplicate-symbol or
// missing-declaration bug in code that is already validated.
#pragma once
// src/artifact/gguf_reader.cpp - P1.S2: GGUF v3 reader, mmap, no ggml dependency.
//
// The C++ counterpart of tools/gguf_reader.py, which has been the reference since P0.S6 (written
// because gguf-py cannot represent type 42 / Q2_0). Per P1.S2 it must:
//   * mmap the file and parse header, metadata KV (ALL value types incl. arrays), tensor directory
//   * be multi-shard aware (split.* keys)
//   * carry an architecture guard: general.architecture == "qwen4exp" and the compiled-in constants
//     must match, refusing with a precise error otherwise
//   * have no ggml dependency
//
// This file is the reader plus a `--check` mode that validates it the way the Python one is
// validated: parse the tiny model AND both real shards, and report counts that other harnesses have
// already established independently (1,223 / 1 tensors; 202 Q2_0 in shard 1). Agreement with those
// numbers is the test.
//
// Build: scripts/build_artifact.bat        Run: gguf_reader.exe <file.gguf> [--check]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stdexcept>
#include <algorithm>

#include "strata/artifact/gguf_split.hpp"
#include "strata/core/layout.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX   // windows.h's min/max macros would break std::min/std::max in every file that includes this one
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata {

// ---- ggml type ids we care about. 42 = Q2_0, the PrismML ternary 2-bit encoding this engine targets.
inline const char* ggml_type_name(uint32_t t) {
    switch (t) {
    case 0:
        return "F32";
    case 1:
        return "F16";
    case 2:
        return "Q4_0";
    case 3:
        return "Q4_1";
    case 6:
        return "Q5_0";
    case 7:
        return "Q5_1";
    case 8:
        return "Q8_0";
    case 9:
        return "Q8_1";
    case 10:
        return "Q2_K";
    case 11:
        return "Q3_K";
    case 12:
        return "Q4_K";
    case 13:
        return "Q5_K";
    case 14:
        return "Q6_K";
    case 15:
        return "Q8_K";
    case 16:
        return "IQ2_XXS";
    case 17:
        return "IQ2_XS";
    case 18:
        return "IQ3_XXS";
    case 19:
        return "IQ1_S";
    case 20:
        return "IQ4_NL";
    case 21:
        return "IQ3_S";
    case 22:
        return "IQ2_S";
    case 23:
        return "IQ4_XS";
    case 24:
        return "I8";
    case 30:
        return "BF16";
    case 34:
        return "TQ1_0";
    case 35:
        return "TQ2_0";
    case 39:
        return "MXFP4";
    case 40:
        return "NVFP4";
    case 41:
        return "Q1_0";
    case 42:
        return "Q2_0";
    default:
        return "?";
    }
}

// Block geometry: (elements per block, bytes per block). Q2_0 is 64/18 - proven from this artifact's
// own offset brackets in P0.S6 and recorded in docs/q2_0-contract.md.
inline bool block_geometry(uint32_t t, int& elems, int& bytes) {
    switch (t) {
    case 0:
        elems = 1;
        bytes = 4;
        return true;
    case 1:
    case 30:
        elems = 1;
        bytes = 2;
        return true;
    case 2:
        elems = 32;
        bytes = 18;
        return true;
    case 3:
        elems = 32;
        bytes = 20;
        return true;
    case 6:
        elems = 32;
        bytes = 22;
        return true;
    case 7:
        elems = 32;
        bytes = 24;
        return true;
    case 8:
        elems = 32;
        bytes = 34;
        return true;
    case 9:
        elems = 32;
        bytes = 36;
        return true;
    case 10:
        elems = 256;
        bytes = 84;
        return true;
    case 11:
        elems = 256;
        bytes = 110;
        return true;
    case 12:
        elems = 256;
        bytes = 144;
        return true;
    case 13:
        elems = 256;
        bytes = 176;
        return true;
    case 14:
        elems = 256;
        bytes = 210;
        return true;
    case 16:
        elems = 256;
        bytes = 66;
        return true;
    case 17:
        elems = 256;
        bytes = 74;
        return true;
    case 18:
        elems = 256;
        bytes = 98;
        return true;
    case 20:
        elems = 32;
        bytes = 18;
        return true;
    case 21:   // IQ3_S
        elems = 256;
        bytes = 110;
        return true;
    case 22:   // IQ2_S
        elems = 256;
        bytes = 82;
        return true;
    case 23:
        elems = 256;
        bytes = 136;
        return true;
    case 29:   // IQ1_M
        elems = 256;
        bytes = 56;
        return true;
    case 24:   // I8: raw bytes (the FP8 PLE table of tools/ple_fp8_pack.py)
        elems = 1;
        bytes = 1;
        return true;
    case 42:
        elems = 64;
        bytes = 18;
        return true;
    default:
        return false;
    }
}

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> shape; // GGUF order: dim 0 varies fastest
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t elements() const {
        uint64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    const char* type_name() const { return ggml_type_name(type); }
};

// A bounds-checked cursor over the mmapped header region. Every read is checked, so a truncated or
// corrupt file produces a precise error rather than a segfault.
class Cursor {
public:
    Cursor(const uint8_t* base, size_t size) : base_(base), size_(size) {}
    void need(size_t n) const {
        if (pos_ + n > size_) throw std::runtime_error("GGUF: unexpected end of file in header");
    }
    template <class T> T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = read<uint64_t>();
        need((size_t)n);
        std::string s(reinterpret_cast<const char*>(base_ + pos_), (size_t)n);
        pos_ += (size_t)n;
        return s;
    }
    size_t pos() const { return pos_; }

private:
    const uint8_t* base_;
    size_t size_;
    size_t pos_ = 0;
};

enum class MetaType : uint32_t { U8 = 0, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 };

struct MetaValue {
    MetaType type = MetaType::U32;
    uint64_t u = 0;                // integer payloads
    double f = 0;                  // float payloads
    std::string s;                 // string payloads
    MetaType elem = MetaType::U32; // arrays
    uint64_t count = 0;
    std::vector<MetaValue> items;
    bool is_num() const { return type != MetaType::STRING && type != MetaType::ARRAY; }
    double num() const { return (type == MetaType::F32 || type == MetaType::F64) ? f : (double)u; }
};

inline MetaValue read_value(Cursor& c, MetaType t, int depth = 0) {
    if (depth > 2) throw std::runtime_error("GGUF: array nesting too deep");
    MetaValue v;
    v.type = t;
    switch (t) {
    case MetaType::U8:
        v.u = c.read<uint8_t>();
        break;
    case MetaType::I8:
        v.u = (uint64_t)(int64_t)c.read<int8_t>();
        break;
    case MetaType::U16:
        v.u = c.read<uint16_t>();
        break;
    case MetaType::I16:
        v.u = (uint64_t)(int64_t)c.read<int16_t>();
        break;
    case MetaType::U32:
        v.u = c.read<uint32_t>();
        break;
    case MetaType::I32:
        v.u = (uint64_t)(int64_t)c.read<int32_t>();
        break;
    case MetaType::F32: {
        float x = c.read<float>();
        v.f = x;
        v.u = 0;
        break;
    }
    case MetaType::BOOL:
        v.u = c.read<uint8_t>() ? 1 : 0;
        break;
    case MetaType::STRING:
        v.s = c.str();
        break;
    case MetaType::U64:
        v.u = c.read<uint64_t>();
        break;
    case MetaType::I64:
        v.u = (uint64_t)c.read<int64_t>();
        break;
    case MetaType::F64:
        v.f = c.read<double>();
        break;
    case MetaType::ARRAY: {
        v.elem = (MetaType)c.read<uint32_t>();
        v.count = c.read<uint64_t>();
        if (v.count > (1u << 24)) throw std::runtime_error("GGUF: implausible array length");
        v.items.reserve((size_t)std::min<uint64_t>(v.count, 64));
        for (uint64_t i = 0; i < v.count; ++i) {
            MetaValue e = read_value(c, v.elem, depth + 1);
            if (i < 64) v.items.push_back(std::move(e)); // keep a sample; the count is what matters
        }
        break;
    }
    default:
        throw std::runtime_error("GGUF: unknown metadata value type");
    }
    return v;
}

class GgufFile {
public:
    explicit GgufFile(const std::string& path) : path_(path) {
        try { open(); }
        catch (...) { close(); throw; }
    }
    ~GgufFile() { close(); }
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string, MetaValue>& metadata() const { return meta_; }
    uint32_t version() const { return version_; }
    uint64_t data_start() const { return data_start_; }
    uint64_t file_size() const { return size_; }
    uint64_t alignment() const { return alignment_; }
    const std::string& path() const { return path_; }

    const TensorInfo* find(const std::string& name) const {
        for (const auto& t : tensors_)
            if (t.name == name) return &t;
        return nullptr;
    }
    uint64_t count_type(const char* tn) const {
        uint64_t n = 0;
        for (const auto& t : tensors_)
            if (std::strcmp(t.type_name(), tn) == 0) ++n;
        return n;
    }
    const MetaValue* get(const std::string& key) const {
        auto it = meta_.find(key);
        return it == meta_.end() ? nullptr : &it->second;
    }
    const uint8_t* tensor_data(const TensorInfo& t) const { return base_ + data_start_ + t.offset; }

private:
    void open() {
#ifdef _WIN32
        HANDLE h = CreateFileA(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path_);
        LARGE_INTEGER li{};
        GetFileSizeEx(h, &li);
        size_ = (uint64_t)li.QuadPart;
        HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!m) {
            CloseHandle(h);
            throw std::runtime_error("CreateFileMapping failed");
        }
        base_ = (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        if (!base_) {
            CloseHandle(m);
            CloseHandle(h);
            throw std::runtime_error("MapViewOfFile failed");
        }
        map_ = m;
        file_ = h;
#else
        int fd = ::open(path_.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path_);
        fd_ = fd;
        struct stat st{};
        if (fstat(fd, &st) != 0) throw std::runtime_error("fstat failed");
        size_ = (uint64_t)st.st_size;
        void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        base_ = (const uint8_t*)p;
#endif
        parse();
    }
    void close() {
#ifdef _WIN32
        if (base_) UnmapViewOfFile(base_);
        if (map_) CloseHandle((HANDLE)map_);
        if (file_) CloseHandle((HANDLE)file_);
#else
        if (base_) munmap((void*)base_, size_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }

    void parse() {
        Cursor c(base_, size_);
        const uint32_t magic = c.read<uint32_t>();
        if (magic != 0x46554747u) throw std::runtime_error("not a GGUF file (bad magic)");
        version_ = c.read<uint32_t>();
        if (version_ != 3)
            throw std::runtime_error("GGUF v" + std::to_string(version_) + ", this reader handles v3");
        const uint64_t n_tensors = c.read<uint64_t>();
        const uint64_t n_kv = c.read<uint64_t>();

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = c.str();
            MetaType t = (MetaType)c.read<uint32_t>();
            meta_.emplace(std::move(key), read_value(c, t));
        }
        tensors_.reserve((size_t)n_tensors);
        // GGUF has no index to arbitrate between two tensors of one name: find() is first-match, so a
        // duplicate would silently win by position.  Refuse the file at open instead, naming both.
        std::set<std::string> names;
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            t.name = c.str();
            if (!names.insert(t.name).second)
                throw std::runtime_error("GGUF: duplicate tensor name '" + t.name + "' in " + path_);
            const uint32_t nd = c.read<uint32_t>();
            if (nd == 0 || nd > 4) throw std::runtime_error("GGUF: bad n_dims for " + t.name);
            t.shape.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) t.shape[d] = c.read<uint64_t>();
            t.type = c.read<uint32_t>();
            t.offset = c.read<uint64_t>();
            tensors_.push_back(std::move(t));
        }
        uint64_t align = 32;
        if (const MetaValue* a = get("general.alignment"))
            if (a->u) align = a->u;
        alignment_ = align;
        data_start_ = (c.pos() + align - 1) / align * align;
        if (data_start_ > size_) throw std::runtime_error("GGUF: data section starts past EOF");
    }

    std::string path_;
    const uint8_t* base_ = nullptr;
    uint64_t size_ = 0, data_start_ = 0, alignment_ = 32;
    uint32_t version_ = 0;
#ifdef _WIN32
    void* file_ = nullptr;
    void* map_ = nullptr;
#else
    int fd_ = -1;
#endif
    std::vector<TensorInfo> tensors_;
    std::map<std::string, MetaValue> meta_;
};

// Bytes of a tensor's payload from its shape and block geometry; 0 when the type is unknown, a row is not whole
// blocks, or the count overflows.
inline uint64_t tensor_payload_bytes(const TensorInfo& t) {
    int be = 0, bb = 0;
    if (t.shape.empty() || !block_geometry(t.type, be, bb) || t.shape[0] % (uint64_t) be) return 0;
    uint64_t elements = 1;
    for (uint64_t d : t.shape) {
        if (d == 0 || elements > (std::numeric_limits<uint64_t>::max)() / d) return 0;
        elements *= d;
    }
    const uint64_t blocks = elements / (uint64_t) be;
    if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) bb) return 0;
    return blocks * (uint64_t) bb;
}

// The shards of one model (strata::gguf_split_paths), opened together; tensors are looked up across all of them.
// From eddoursul/Strata 8029fa9, with the split-key validation of #255 (gopinath87607) made a property of the
// model rather than of one loader: a split GGUF carries the model's metadata (general.architecture and the
// qwen4exp keys) in its FIRST shard only, and the later shards just declare split.count / split.no /
// split.tensors.count.  Unsloth's UD-Q4_K_XL is the extreme case: shard 1 holds the metadata and no tensor at
// all, and output.weight, token_embd.weight and the PLE table live in shard 2.  So the architecture is checked
// on `meta()` and a tensor is read from the shard that holds it.
//
// Refused at construction: a later shard whose split keys disagree with shard 1's (another model's shard, or a
// shard renamed into the family), a split model whose tensor directories do not add up to split.tensors.count,
// and a tensor name present in two shards (GGUF has no index to say which one is meant).
class GgufModel {
public:
    explicit GgufModel(const std::vector<std::string>& paths) {
        if (paths.empty()) throw std::runtime_error("GGUF: a model needs at least one shard");
        for (const auto& p : paths) shards_.push_back(std::make_unique<GgufFile>(p));
        validate_split();
        for (size_t i = 0; i < shards_.size(); ++i)
            for (const auto& t : shards_[i]->tensors()) {
                const auto ins = index_.emplace(t.name, std::make_pair(i, &t));
                if (!ins.second)
                    throw std::runtime_error("GGUF: tensor " + t.name + " is in two shards (" +
                                             shards_[ins.first->second.first]->path() + " and " +
                                             shards_[i]->path() + ")");
            }
    }
    /// Opens every shard of the model that `any_shard` belongs to (throws when one is missing).
    static GgufModel open(const std::string& any_shard) { return GgufModel(gguf_split_paths(any_shard)); }

    size_t size() const { return shards_.size(); }
    const GgufFile& shard(size_t i) const { return *shards_[i]; }
    /// The metadata shard: general.architecture and the model's keys.
    const GgufFile& meta() const { return *shards_[0]; }
    /// The tensor named `name` (its shard index in `*shard`), or nullptr.
    const TensorInfo* find(const std::string& name, size_t* shard = nullptr) const {
        const auto it = index_.find(name);
        if (it == index_.end()) return nullptr;
        if (shard) *shard = it->second.first;
        return it->second.second;
    }
    /// Whether `t` (a tensor of shard `s`) has a known byte count that lies inside its file.
    bool in_bounds(const TensorInfo& t, size_t s) const {
        const GgufFile& g = *shards_[s];
        const uint64_t bytes = tensor_payload_bytes(t);
        const uint64_t payload = g.file_size() - g.data_start();
        return bytes != 0 && t.offset <= payload && bytes <= payload - t.offset;
    }

private:
    void validate_split() const {
        const size_t n = shards_.size();
        const MetaValue* count0 = shards_[0]->get("split.count");
        if (n == 1) {
            if (count0 && count0->u > 1)
                throw std::runtime_error("GGUF: " + shards_[0]->path() + " is shard 1 of " + std::to_string(count0->u) +
                                         ", but it was opened as a whole model");
            return;
        }
        if (!shards_[0]->get("general.architecture"))
            throw std::runtime_error("GGUF: " + shards_[0]->path() + " has no general.architecture; the first shard "
                                     "of a split model carries the metadata");
        const MetaValue* total = shards_[0]->get("split.tensors.count");
        uint64_t tensors = 0;
        for (size_t i = 0; i < n; ++i) {
            const GgufFile& g = *shards_[i];
            const MetaValue* count = g.get("split.count");
            const MetaValue* no = g.get("split.no");
            const MetaValue* tc = g.get("split.tensors.count");
            if (!count || !no || count->u != n || no->u != i || (total && (!tc || tc->u != total->u)))
                throw std::runtime_error("GGUF: " + g.path() + " does not declare itself shard " + std::to_string(i + 1) +
                                         " of " + std::to_string(n) + " of this model (split.count / split.no / "
                                         "split.tensors.count)");
            tensors += g.tensors().size();
        }
        if (total && tensors != total->u)
            throw std::runtime_error("GGUF: the " + std::to_string(n) + " shards hold " + std::to_string(tensors) +
                                     " tensors, but split.tensors.count is " + std::to_string(total->u));
    }

    std::vector<std::unique_ptr<GgufFile>> shards_;
    std::map<std::string, std::pair<size_t, const TensorInfo*>> index_;
};

// ---- architecture guard (P1.S2). The engine is specialised to ONE model; anything else must be
// refused with a precise error rather than silently mis-run.
// **EVERY FIELD DEFAULTS TO 0, WHICH MEANS "presence only" — DO NOT PUT Flash-Next's SHAPE HERE.**  This struct
// used to carry `48 / 2560 / 24 / 2` as defaults, so calling `check_architecture(g)` with no arguments
// asserted the file WAS Flash-Next: a qwen35moe file (40 layers) was refused as "block_count = 40, expected 48",
// and — worse — any file that happened to match those numbers would have been accepted as a different model.
// A default that names one model inside a guard for all models is not a default, it is an assumption.
// Callers that know the shape they want pass it; callers that do not get a presence check, which is the
// honest answer for "I don't know yet" (and `read_geometry` is what turns presence into numbers).
struct Qwen4ExpGuard {
    uint32_t block_count = 0, hidden = 0, experts = 0, experts_used = 0, head_count = 0, head_count_kv = 0;
};

// ---- architecture guard (P1.S2). The engine is specialised to ONE model; anything else must be
// refused with a precise error rather than silently mis-run.
//
// **A GUARD IS A WHITELIST, SO A NEW ARCHITECTURE HAS TO BE ADDED OR IT IS REFUSED — which is the point.**
// `qwen35moe` (Qwen3.6-35B-A3B) is the second supported family.  Its tensors are named IDENTICALLY to the ones
// `qwen4exp` uses — `blk.N.ffn_gate_inp.weight`, `ffn_gate_up_exps`, `ffn_*_shexp` — because both come out of the
// same llama.cpp converter lineage.  So no tensor lookup changes; only the metadata KEY PREFIX and the set of
// shapes that must be checked differ.
//
// What differs, and is why this is not just a string swap:
//   - n_embd 2048 (not 2560), 40 layers (not 48), 16 heads (not 24), 2 KV heads (same)
//   - 256 experts (not 512), top-8 (not 10)
//   - shared expert width 512 == its per-expert width; Flash-Next's shared expert is a different number
//   - no hyper-connections (hc) and no indexer at all — handled by the geometry, not here
//
// `prefix` is the arch string itself, because llama.cpp namespaces every key under `<arch>.`.
inline std::string check_architecture(const GgufFile& g, const Qwen4ExpGuard& want = {}) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "qwen4exp" && arch->s != "qwen35moe")
        return "architecture is '" + arch->s + "', this engine requires 'qwen4exp' or 'qwen35moe'";
    // The keys below are arch-prefixed, so a qwen35moe file must be asked for ITS prefix or every lookup misses.
    const std::string p = std::string(arch->s) + ".";
    struct Req {
        const char* key;
        uint64_t want;
    };
    const Req reqs[] = {
        {"block_count", want.block_count},
        {"embedding_length", want.hidden},
        {"expert_count", want.experts},
        {"expert_used_count", want.experts_used},
        {"attention.head_count", want.head_count},
        {"attention.head_count_kv", want.head_count_kv},
    };
    std::string missing;
    for (const auto& r : reqs) {
        const MetaValue* v = g.get(p + r.key);
        if (!v) {
            // A qwen35moe file has no `expert_used_count` when every expert is used, and a dense pack has neither.
            // Only the two shapes the graph cannot guess are load-bearing; the rest are presence checks, and a
            // missing one is reported rather than defaulted so a truncated shard cannot masquerade as a model.
            if (!missing.empty()) missing += ", ";
            missing += p + r.key;
            continue;
        }
        if (r.want && v->u != r.want)
            return p + r.key + " = " + std::to_string(v->u) + ", expected " + std::to_string(r.want);
    }
    if (!missing.empty()) return "missing " + missing;
    return {}; // empty == ok
}

/// **READ THE WHOLE GEOMETRY OUT OF THE MODEL FILE, INSTEAD OF LEAVING Flash-Next's CONSTANTS BEHIND.**
///
/// This is stage 3's deliverable.  Until now `generate.cpp` read only `expert_count` and `expert_used_count`
/// and left `n_embd = 2560`, `n_layers = 48`, `n_head = 24`, `n_expert = 512` etc. as compile-time defaults, so
/// pointing Strata at a Qwen3.6-35B file would have sized every buffer from Flash-Next's shape.  Those numbers
/// load silently: an arena that is 25% too big is still an arena, and the first symptom is a plausible wrong
/// number rather than a refusal.
///
/// `ModelGeometry` is in `strata/core/layout.hpp`, which does not know about GGUF, so this lives on the reader
/// side and returns by value.  Every field that both families have is READ, never defaulted: a geometry field left
/// at Flash-Next's value is exactly the bug this function exists to remove, and a missing key is far more
/// likely to be a wrong model file than an intentional default.
///
/// Fields that only one family has are set to the "absent" value (`0`), which is what `has_hc()`,
/// `has_indexer()`, `has_moe()` and `ffn_width()` already test.  So absence is expressed once, here, in the
/// numbers — no second set of flags that can disagree with them.
inline bool read_geometry(const GgufFile& g, core::ModelGeometry& out, std::string& err) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) { err = "missing general.architecture"; return false; }
    const std::string p = std::string(arch->s) + ".";
    // ponytail: everything is READ, nothing is defaulted except where llama.cpp itself defaults it.  A geometry
    // field left at Flash-Next's value is precisely the bug this function exists to remove, and a missing key is
    // far likelier to be a wrong model file than a deliberate default.
    //
    // Every spelling below is taken from llama-arch.cpp's `"%s.<name>"` table, not guessed.
    struct Num { const char* key; int64_t* dst; };
    const Num nums[] = {
        {"embedding_length",               &out.n_embd},
        {"block_count",                    &out.n_layers},
        {"attention.head_count",           &out.n_head},
        {"attention.head_count_kv",        &out.n_head_kv},
        {"attention.key_length",           &out.head_dim},        // optional; derived below
        {"full_attention_interval",        &out.qsa_interval},
        {"expert_count",                   &out.n_expert},
        {"expert_shared_feed_forward_length", &out.n_ff},          // shared width; the routed one is derived
        {"ssm.state_size",                 &out.ssm_state_size},
        {"ssm.group_count",                &out.ssm_k_heads},      // KEY heads
        {"ssm.time_step_rank",             &out.ssm_v_heads},      // VALUE heads
        {"ssm.conv_kernel",                &out.ssm_d_conv},
    };
    std::string missing;
    auto geti = [&](const char* key, int64_t& dst) {
        if (const MetaValue* v = g.get(p + key)) { dst = (int64_t) v->u; return true; }
        return false;
    };
    // **ZERO EVERY OPTIONAL SLOT FIRST.**  `out` arrives holding Flash-Next's defaults, so "the key was absent"
    // and "the key said 256" are the same number unless the slot is cleared.  head_dim is the one that bites:
    // left at 256 it is positive, so the derivation below never runs and a model with no key_length silently
    // gets Flash-Next's head width.  Clear it here rather than reasoning about it at each use.
    out.head_dim = 0;
    for (const auto& n : nums) {
        if (geti(n.key, *n.dst)) continue;
        if (n.dst == &out.head_dim) continue;                 // optional: derived from n_embd / n_head below
        if (!missing.empty()) missing += ", ";
        missing += p + n.key;
    }
    if (!missing.empty()) { err = "geometry is missing " + missing; return false; }
    if (out.head_dim <= 0) out.head_dim = out.n_head ? out.n_embd / out.n_head : 0;   // llama.cpp's own rule
    if (out.head_dim <= 0) { err = "cannot derive head_dim: no head_count and no key_length"; return false; }

    // **THE ROUTED EXPERT WIDTH IS `n_ff / n_expert_used`, NOT A KEY OF ITS OWN.**  A qwen35moe file only carries
    // the shared expert's width (512) and, when it is not itself a scalar, the routed one.  llama.cpp computes the
    // rest exactly here; copying its rule is what keeps the two in step.  Using the shared width instead would be
    // wrong by a factor of `expert_used_count` on every MoE pack whose two widths differ.
    if (const MetaValue* v = g.get(p + "expert_feed_forward_length")) out.n_ff = (int64_t) v->u;
    else if (const MetaValue* v2 = g.get(p + "expert_used_count")) {
        const int64_t used = (int64_t) v2->u;
        if (used <= 0) { err = p + "expert_used_count = " + std::to_string(used) + " (must be > 0)"; return false; }
        out.n_ff = out.n_ff / used;
    }
    if (out.has_moe() && out.n_ff <= 0) { err = "MoE pack has no expert feed-forward width"; return false; }

    // The two SSM derived widths.  Checked against both families: Flash-Next 128*48 = 6144 and
    // 128*(2*16+48) = 10240, Qwen3.6-35B 128*32 = 4096 and 128*(2*16+32) = 8192.
    out.ssm_value_dim = out.ssm_state_size * out.ssm_v_heads;
    out.ssm_conv_channels = out.ssm_state_size * (2 * out.ssm_k_heads + out.ssm_v_heads);

    // Subsystems this family does not have.  Absence is expressed as the zero that `has_hc()`, `has_indexer()`
    // and `ffn_width()` already test, so there is no second set of flags that could disagree with the numbers.
    out.hc = 0;
    out.hc_lr = 0;
    out.idx_q_heads = 0;
    out.idx_key_dim = 0;
    out.ple_ngram_size = 0;
    out.dense_ffn = 0;
    err.clear();
    return true;
}

} // namespace strata
