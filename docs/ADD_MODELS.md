# Strata 加新模型：Qwen3.8-27B / Qwen3.6-35B-A3B 改造清单

> 基于 Strata main 分支（2026-10 拉取，852 文件）。配置数据来自 hf-mirror 的 `config.json`，
> 不是凭记忆。几何映射用 `tools/geometry_from_config.py` 生成，自检 `python tools/geometry_from_config.py`。
>
> **结论先说**：Strata 不是通用推理框架，是一个为 `Qwen3.8-Flash-Next`（`model_type: qwen4_exp_text`）
> 写死的引擎。`include/strata/core/layout.hpp` 的 `ModelGeometry` 是唯一几何来源，
> `src/core/layout.cpp::check_layer()` 在**加载时**逐张量断言名字+shape+量化格式。
>
> 好消息：这两个目标模型都比 Flash-Next **少**两套子系统（gated residual、QSA indexer、PLE n-gram），
> 所以工作量是「删路径 + 换几何」，不是「写新引擎」。

---

## 1. 三方几何对照

数据来源：`hf-mirror.com/Qwen/<repo>/raw/main/config.json` → `text_config`。

| 字段 | Flash-Next（Strata 已支持） | Qwen3.8-27B | Qwen3.6-35B-A3B |
|---|---|---|---|
| `model_type` | `qwen4_exp_text` | `qwen3_5_text` | `qwen3_5_moe_text` |
| `architectures` | — | `Qwen3_5ForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` |
| hidden_size (n_embd) | 2560 | **5120** | **2048** |
| num_hidden_layers | 48 | **64** | **40** |
| full_attention_interval | 4 | 4 | 4 |
| 层分布（linear/full） | 36 / 12 | **48 / 16** | **30 / 10** |
| num_attention_heads | 24 | 24 | **16** |
| num_key_value_heads | 2 | **4** | 2 |
| head_dim | 256 | 256 | 256 |
| linear_num_key_heads | 16 | 16 | 16 |
| linear_num_value_heads | 48 | 48 | **32** |
| linear_value_head_dim (state) | 128 | 128 | 128 |
| linear_conv_kernel_dim | 4 | 4 | 4 |
| num_experts | 512 | **无（稠密）** | **256** |
| num_experts_per_tok | 10 | — | **8** |
| moe_intermediate_size (n_ff) | 640 | —（用 intermediate_size 17408） | **512** |
| shared_expert_intermediate_size | 640 | — | **512** |
| hc_count / hc_lowrank | 4 / 320 | **无** | **无** |
| indexer_n_heads / head_dim | 4 / 128 | **无** | **无** |
| ngram_size (PLE) | 3 | **无** | **无** |
| attn_output_gate | 无显式字段 | **True** | **True** |
| output_gate_type | sigmoid | **swish** | 无 |
| vocab_size | 248320 | 248320 | 248320 |
| max_position_embeddings | 262144 | 262144 | 262144 |
| mtp_num_hidden_layers | 1 | 1 | 1 |
| partial_rotary_factor | 0.25 | 0.25 | 0.25 |
| vision_config | 有（27 层，1152） | 有（out_hidden_size **5120**） | 有（out_hidden_size **2048**） |

三个模型**都是多模态**（`image_token_id` / `video_token_id` / `vision_config`），
GDN 线性注意力 + 每 4 层一次全注意力 + MTP 投机解码 —— 这一层 Strata 的实现**可以复用**。

---

## 2. 架构耦合点在哪（实测扫描，非估计）

按子系统统计全仓命中（排除 `third_party/ sycl/ bench/`）：

| 子系统 | 命中 | 热点文件 |
|---|---:|---|
| MoE experts | 1537 | `src/program/generate.cpp`(269)、`src/core/expert_source.cpp`(148)、`src/kernels/cuda/router_top10.cu`(94)、`src/core/expert_cache.cpp`(45)、`src/prefill/moe_fused.cu`(36) |
| QSA / indexer | 1160 | `src/core/layer.cpp`(97)、`src/prefill/prefill.cpp`(67)、`src/kernels/cuda/qsa.cu`(90)、`src/core/session.cpp`(80)、`src/kernels/qsa_parity.cpp`(85) |
| GDN / linear attn | 1112 | `src/core/layer.cpp`(160)、`src/core/verify.cpp`(106)、`src/prefill/kernels.cu`(68)、`src/prefill/prefill.cpp`(50) |
| PLE / ngram | 658 | `src/program/generate.cpp`(102)、`src/prefill/prefill.cpp`(82)、`src/kernels/ple_parity.cpp`(84)、`src/kernels/cuda/ple.cu`(31) |
| gated residual (hc_/gr_) | 487 | `src/kernels/cuda/gr.cu`(77)、`src/kernels/cuda/fused_gr.cu`(40)、`src/core/layer.cpp`(28)、`src/kernels/gr_parity.cpp`(65) |
| shared_expert | 156 | `src/kernels/cuda/shared_expert.cu`(24) |

`ModelGeometry` 出现在 18 个文件、共 200+ 处引用。

硬编码常量：

| 常量 | 位置 | 影响 |
|---|---|---|
| `n_expert = 512` | `include/strata/core/layout.hpp:47`（唯一一处定义） | 好消息：几何已经是 struct，改默认值即可 |
| `24576`（48×512 专家总数） | `src/program/generate.cpp`、`src/kernels/decode_cluster_parity.cpp` | 要改成 `n_layers × n_expert` 计算 |
| top-10 router | `src/kernels/cuda/router_top10.cu`(28处)、`include/strata/kernels/router_top10.hpp` | 函数签名已带 `n_expert`/`k` 参数，**但文件名和 kernel 名写死 top10**；内部固定 10 专家布局需确认 |
| `2560` | 69 个文件 229 处 | 多数是 kernel 里的 tile 常量或测试数据，**逐个核对**，不能全局替换 |

---

## 3. 分阶段改造

### 阶段 0：几何参数化（半天，机械）— **已完成**

`ModelGeometry` 本来就是 struct，`check_layer()` 本来就按它断言。所以：

```bash
python tools/geometry_from_config.py bench/q38_27b.json   # 打印可粘贴的 C++ 块 + 待办清单
python tools/geometry_from_config.py bench/q36_35b.json
```

**实际改动**（commit `3b8dbea`，2026-10-05）：

| 文件 | 改了什么 |
|---|---|
| `include/strata/core/layout.hpp` | 加 `ple_ngram_size`（默认 3）+ `has_moe()` / `has_hc()` / `has_indexer()` / `has_ple()` / `n_experts_total()`。四个谓词全是对既有字段的比较 —— 不引入第二套能和几何不一致的状态 |
| `src/core/layout.cpp` | `Want2`/`Want1` 的两个 bool（`qsa_only`,`gdn_only`）换成两个独立门 `WantSub`（pack 有没有这个子系统）× `WantLayer`（这层是不是这种）。**indexer 需要两个门同时开**，原来只有层级门 |
| `include/strata/core/weights.hpp` | `find()` 移到头文件 inline（3 行）；加 `insert()` 给测试搭表 |
| `src/core/weights.cpp` | 删掉搬走的 `find()` |
| `tests/core/layout_subsystem_test.cpp` | 新增：三种架构各一份手搭 `WeightTable`，纯 CPU |

验证：`g++ -std=c++17 -Wall -Wextra` 编译无警告 → `all cases pass`；
**7 处变异测试 7/7 被抓住**（`has_indexer`/`has_moe`/`has_ple` 恒真、`n_experts_total` 写死 24576、
indexer 丢层级门、hc 丢子系统门、MoE router 丢子系统门）。

关键防退化断言（**闸门不能只是不再发问**）：
- qwen3.6-35b 去掉 router → `check_all` 必须**仍然拒绝**
- 稠密 pack 把 `n_expert` 改成 8 → `check_all` 必须**仍然拒绝**

编译方式（无需 CUDA，本机 Windows 无 nvcc，走 WSL）：

```bash
ln -sfn "/mnt/e/zip/agent file big/01_项目代码/Strata" /opt/src/strata   # 中文路径过 WSL 会被编码吃掉
wsl -d debian-bookworm -- bash -c 'cd /opt/src/strata && \
  g++ -std=c++17 -Iinclude -o /tmp/t tests/core/layout_subsystem_test.cpp src/core/layout.cpp && /tmp/t'
```

> ponytail: 引擎里没有引入 JSON 解析器 —— `weights.hpp` 头部明确禁掉了（"a JSON parser in the engine
> would be a new, unaudited component whose failure mode is a wrong byte offset"）。几何仍由几何字段表达。

> **阶段 0 里我说错的一条**：之前报告「`24576` 有 2 处硬编码要改成 `n_layers*n_expert`」—— 实际核查，
> `generate.cpp:3572` 是注释，`decode_cluster_parity.cpp:412` 是 argmax 测试的词表，**都不是硬编码**。
> 真正的专家总数计算在 `expert_source.cpp` 等处已经走 `g.n_expert`。已加 `n_experts_total()` 作为
> 唯一的推导入口，替换散落的字面量。

### 阶段 1：干掉 QSA indexer（2–3 天）— **已完成，不需要新 kernel**

> 原计划这里写着「2–3 天」。实际 **1 小时**，因为原计划假设要写新 kernel。不需要。

`qsa_layer` 里 indexer 只干一件事：**算出 top-k 的 cell id**。没有 indexer 的 pack 需要全选。
而 `qsa_selection_width(n_kv, s) = min(n_kv, idx_top_k + idx_block - 1)` —— 只要 `idx_top_k` 抬到
`kTopkMaxCells`，全选**自动成立**。`qsa_index_step` + `topk_512` + `kv_gather` + `qsa_attend`
一行不改就是稠密注意力。

```cpp
// include/strata/kernels/qsa.hpp，紧挨 QsaShapes
inline QsaShapes qsa_shapes(const strata::core::ModelGeometry& g) {
    QsaShapes s = qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads; s.idx_dim = g.idx_key_dim;
    if (!g.has_indexer()) s.idx_top_k = kTopkMaxCells;   // ← 全部行为改动就这一行
    return s;
}
```

**顺带修掉一个真 bug**：旧代码无条件复制 `idx_top_k = 2048`。一个没有 indexer 的 pack 在 10K 上下文上
会被静默截断成 2051 个 cell —— **输出有限、结果错、不报错**。测试里直接断言了这个数字。

**顺带收掉 11 份重复**：这段算术此前抄在 `layer/mtp/session/verify/prefill(×4)/generate`，
CUDA 侧 7 份 + `sycl/`(HIP) 侧 6 份。现在 `qsa_shapes` 是唯一定义。`qsa.hpp` 加了
`#include "strata/core/layout.hpp"` —— 之前只 include `<cstdint>`，而 `layout.hpp`/`weights.hpp`
都不碰 CUDA，所以测试纯 CPU 可编译。

CUDA 侧和 sycl 侧一起改：只改一边会让 AMD 构建保留截断行为。

**`ponytail:`** 每 token 仍跑一次全量 top-512 排序 —— 明知故犯的 O(n_kv log n_kv) 浪费，
因为我们本来就知道答案是全选。换 decode 速度真的不够时，用 device 端 `iota` 直喂 `kv_gather`。
上限仍是 `kTopkMaxCells = 32768`，`topk_512` 会明确报错而不是截断。

测试：`tests/kernels/qsa_shapes_test.cpp`，15 个断言，纯 CPU。
验证：`-Wall -Wextra` 无警告；5 处变异（删分支/取反/写死 2048/不取 n_head/预算减半）全部 `exit=1`。

commit `9e1a9fd`。147 插入 / 90 删除。

阶段 0 已经把 `Want2[]`/`Want1[]` 的子系统门做好了，所以原来这条待办
「把 `indexer.q_proj` 标成可选」**已经完成**，不需要 `bool required`。剩下的是：

- `qsa.cu` / `qsa_select.cu` / `qsa_prompt_attn.cu` 对这两个 pack 是死代码 → 用 `#if` 或运行时分支跳过
- `src/plan/plan.hpp` 的显存预算公式要重算（没有 indexer key store，但多了 full attention 的 KV）
- **`check_layer` 的 `hc_*` / `ngram_*` 门控阶段 0 已一起做完**（`WantSub::kHc` / `has_ple()`）

**省的部分**：`check_layer` 里那些「indexer.head_count 计数错」的注释说明这块代码本身有历史 bug，
少一套少一份风险。

### 阶段 2：干掉 gated residual（1 天）— **完成**

`888d8bc`（`06a0240` 预算 + `888d8bc` kernel 路径）

**我上一轮写在本文档里的推导是错的，已更正。** 当时写"hc=1 时逐流 RMSNorm + mean 塌缩成一次
普通 RMSNorm，`2*sigmoid` 门恒等于 1，所以 `gr_write` 塌缩成 `R += block_out`"——听起来自洽，
但那是**推出来的**，不是读出来的。读 `gr_parity.cpp:70` 的 CPU oracle 才看清：它有**两处除以 hc**：

```cpp
lo[k]    = silu(p / hc)          // L104
mixed[d] = m / hc                // L125
w[c]     = 2*sigmoid(inject[c] / hc)   // gr.hpp 文档，gr_write
```

所以 `hc = 0` 给的是 `mixed = 0/0` 和 `2*sigmoid(inject/0)` —— **NaN，不是普通残差**，
而且不报错。`2*sigmoid` 确实以 1 为中心（零注入 = 纯加法），但注入是 `bf16(xn) @ w_inject.T`
对着一组**并不存在**的权重，不是零。hc=1 也不会塌缩成上面那个样子。

**正确的退化形态来自 manifest，不是推导。** 查 Qwen3.6-35B 的 `model.safetensors.index.json`，
`layers.0` 的张量里有：

```
input_layernorm.weight
post_attention_layernorm.weight
```

没有 `hc_*`。而 Flash-Next 里正是 `hc_attn_norm` / `hc_ffn_norm` 扮演这两个角色。所以无 hc 时：

| | 有 hc（Flash-Next） | 无 hc（35B / 27B） |
|---|---|---|
| mixer | `hc_*_norm` 归一化 + 门控混合 | `input_layernorm` / `post_attention_layernorm` |
| residual | `hc * n_embd` 的栈 | `n_embd` 的向量 |
| 写回 | `R + block_out * 2*sigmoid(inject/hc)` | `R + block_out` |

**不需要新 kernel，两个已存在的调用就够：**

```cpp
copy_from_mapped(bb.mixed, R, g.n_embd, stream);
rms_norm_weighted(bb.mixed, w_norm, 1, g.n_embd, RMS_EPS, stream);   // mixer
strata::kernels::add_inplace(bb.R, bb.block_out, g.n_embd, stream);    // 残差
```

`add_inplace` 是逐元素 `dst[i] += src[i]`，所以 `dst == src` 安全（每个线程只碰自己的 i）——
这正是原地 `R += block_out`。

> **第一版漏了缓冲区宽度，那是最严重的一处。** `b.R` 按 `g.hc * g.n_embd` 分配，`hc = 0` 时是
> **0 floats** —— 而无 hc 的块仍然要做 `R += block_out`，整个普通 Transformer 残差写在缓冲区
> 起点之外。不崩溃，只是静默损坏。`block_buffers_bytes` 和 `block_buffers_init` 都改成 `has_hc()`
> 三元。判据是 `has_hc()` 而不是 `hc != 0`：那才是加载器检查过的东西，且 `hc = 1` 仍然是一栈。

> 踩到自己的两个错：① 先写成 `rms_norm_weighted(R, ...)` 就地归一化 `R` —— 那会毁掉本块
> 还没写回的残差；`gr_read` 从不改 `R` 正是这个原因。② 裸调 `add_inplace` / `rms_norm_weighted`
> 编译不过 —— 它们在 `strata::kernels`，匿名 namespace 里有 using 声明，`strata::core` 里没有。

测试 `tests/kernels/gr_degenerate_test.cpp`（纯 CPU，20 断言）。**变异 8/8 被抓住**，
包括"恢复成 `g.hc * g.n_embd`"（我实际写出去的那个 bug）。
完整 CUDA 构建 225/225 通过（nvcc 12.9, sm_86），四个 CPU 测试全绿。

> 早先记的"本机没有 nvcc，编不了"已作废：见 [BUILD_ENV.md](BUILD_ENV.md)。WSL + CUDA 12.9 已装好并实测。

注意：`gr_*` 在 `prefill/kernels.cu` 里也占 10 处、`ple.cu` 占 45 处 —— 说明 **PLE 和 GR 在
prompt 路径里是耦合的**，动 GR 要连带看 PLE。

### 阶段 3：MoE 参数化（Qwen3.6-35B-A3B 专用）— **代码完成，未做真实权重推理**

#### 先纠正一个我一开始搞错的对比

我先拿 safetensors 名字（`mlp.experts.gate_up_proj`、`mlp.gate.weight`）去比 Strata 期待的
`ffn_gate_up_exps`、`ffn_gate_inp`，得出"张量命名完全不同，阶段 3 是一份重命名工程"。

**这个对比本身是错的。** Strata 读的是 GGUF，不是 safetensors；而 llama.cpp 的 `qwen35moe` GGUF 里
张量名**和 Flash-Next 完全一致**（`llama.cpp/src/models/qwen35moe.cpp` 里就是
`ffn_gate_inp` / `ffn_gate_up_exps` / `ffn_down_exps` / `ffn_*_shexp`）—— 同一个 converter 血统。

所以阶段 3 **不需要新 kernel、不需要重命名层**。真正要做的只有两件，都是"元数据"层面的。

#### 1. 架构门槛

`check_architecture` 只认 `qwen4exp`。加白名单 `qwen35moe`（`llama-arch.cpp` L41/L42 的字符串）。

**键前缀必须跟着 arch 走。** 原来硬编码 `qwen4exp.` 前缀，所以一个 qwen35moe 文件会"成功"地
读不到任何键，返回空 —— 每个字段都沿用 Flash-Next 的默认值。

#### 2. 几何从文件读，不留编译期常量

原来 `generate.cpp` 只读 `expert_count` 和 `expert_used_count`，`n_embd = 2560`、`n_layers = 48`、
`n_head = 24` 全是常量。新增 `strata::read_geometry()`（在 reader 侧，因为 `layout.hpp` 不认识 GGUF）。

**这个 bug 的形状是"静默"的**：arena 大 25% 仍然是个 arena，第一个症状是一个看起来合理的错数字，
而不是拒绝。缺必需键现在报错，不默认；只有 `head_dim` 可选（按 llama.cpp 的规则推
`n_embd / n_head`）。

#### 顺手抓到的两个真 bug

**(a) `Qwen4ExpGuard{}` 的默认值就是 Flash-Next 的形状**（48/2560/24/2）。`check_architecture(g)`
无参调用会断言"这文件是 Flash-Next"—— 40 层的 qwen35moe 被拒为 `block_count = 40, expected 48`。
一个给所有模型用的 guard 里塞某个模型的默认值，不是默认值，是假设。改成全 0 = presence-only。

**(b) `read_geometry` 里 `head_dim` 没先清零。** `out` 带着 Flash-Next 的默认值进来，key_length
缺失时 `head_dim` 留在 256，是正数，于是"键缺失"和"文件说 256"变成同一个数，推导分支永不执行，
模型静默拿到 Flash-Next 的 head 宽度。先清零再读。

#### 还有一个宽度陷阱

文件里只有 shared expert 的 `expert_shared_feed_forward_length`（512），**路由专家宽度必须
`512 / expert_used_count = 64`**（llama.cpp 就是这么算的）。直接拿 512 会让每个专家宽 8 倍。

#### SSM 两个派生宽度

`ssm_value_dim = state_size × v_heads`、`ssm_conv_channels = state_size × (2×k_heads + v_heads)`。
两个家族都对得上：Flash-Next 128×48=6144 / 128×80=10240，Qwen3.6-35B 128×32=4096 / 128×64=8192。
注意 conv 是 `2*k + v`：qwen35moe 里 `key_dim*2 + value_dim` 就是这么写的。

#### 键名的教训

我第一版写的 `hyper_connections` 是**猜的**，真名是 `hyper_connection.count` /
`hyper_connection.low_rank`。所有键名现在都取自 `llama-arch.cpp` 的 `"%s.<name>"` 表，一张表
174 个键，全部列出来对着挑。

#### 验证

- 完整 CUDA 构建 56/56，0 error；`strata-device --selftest` 通过
- 6 个 CPU 测试全绿（新增 `geometry_reader_test`，写一个只有元数据的最小 GGUF 让 `read_geometry` 真跑）
- **6 类变异全被抓住**（含上面两个 bug、前缀写死、门槛拒绝 qwen35moe、hc 不清零）

写测试的 GGUF 写入器连错四次，每次报错都指向一个不同的真实约束，值得记下来：
1. magic 必须是 `0x46554747`（"GGUF" 小端）
2. 头是 **tensors count 在前、kv count 在后**（我一开始写反了）
3. `MetaType::U32` 的 type id 是 **4**（enum 里 U8=0 I8 U16 I16 U32）。我一度"修正"成 6，那是 F32 ——
   改错之后**所有键都读成 0**，看起来像逻辑 bug，其实是 type id 错位
4. data_start 默认对齐是 **32**，不是 8

**`Geometry` 里那 20 个字段一个都没参与本阶段测试** —— 它们全在 `dense_ffn`/`n_ff` 之外，
真正的 MoE 数值路径（router top-8、专家 GEMV、combine）仍然只有 Flash-Next 的实现在跑。

#### 阶段 3 剩下的真活（已实测，不是估计）

**1. prompt 路径的 289 处编译期几何常量。** `src/prefill/prefill.cpp:90`:

```cpp
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
```

逐一数过，共 **289 处**引用，其中 9 个常量全是模型几何：

| 常量 | 是什么 | 引用数 |
|---|---|---|
| N | n_embd | 113 |
| K | top-k | 76 |
| D | n_embd × hc | 32 |
| ZV | ssm_value_dim | 15 |
| LR | hc_lr | 15 |
| HC | hc | 12 |
| HV | ssm_v_heads | 10 |
| C | ssm_conv_channels | 9 |
| NE | n_expert | 7 |

**好消息**：它不是静默出错。L737 有一道门：

```cpp
if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
    err = "prefill: geometry differs from the artifact's"; return false;
}
```

所以给一个 qwen35moe 文件跑 prefill 会**明确报错**，而不是算出垃圾。这一点和阶段 4 那个 27 倍
静默越界不一样 —— 这里的作者留了门。

**坏消息**：门后面 289 处要改，其中两处卡在编译期：

```cpp
static constexpr int kGrpEv = NE / 16 + 2;      // L397
static constexpr size_t kHostBounds = 2 * (NE + NE / 16 + 2) + 64;   // L446
```

这两处是 `static constexpr`，要变成运行时成员才能跟着 `n_expert` 变。

注意 `NE / 16` 这个 16 是 `MMQ_GROUP`（分组宽度），**Qwen3.6-35B 的 256 个专家要 16 个组**，
而 Flash-Next 的 512 也是 16 组 —— 巧合相等，但不是因为设计如此。改的时候要确认 `MMQ_GROUP`
本身是不是也该参数化（它现在 42 处引用，是个调优旋钮，未必是几何）。

**2. decode 路径已有守卫但也有硬编码**（`src/core/layer.cpp` / `verify.cpp`）：

```cpp
native_router_enabled() && g.n_expert == 512 && k == 10      // layer.cpp:370
native_router_enabled() && NE == 512 && K == 10              // verify.cpp:909
```

这两处是 **guarded fast path**：不匹配就走通用路径，所以是安全的（不匹配 = 慢，不是错）。
但要确认 256 专家 / top-8 在通用路径上真的能跑，而不是只在这些 fast path 里被测过。

**3. `mtp`（Qwen3.6-35B 带 `mtp.layers.0`，`linear_fc1`/`linear_fc2`）完全没碰。**

**4. 没有加载过任何 qwen35moe 权重。** 阶段 5（GGUF 量化）之前拿不到真文件。

。