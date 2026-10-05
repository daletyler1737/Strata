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

### 阶段 2：干掉 gated residual（1 天）— **一半完成**

拆成两半：

**已完成：显存预算（commit `06a0240`，+20 行）**

`plan::Geometry` 是同一段算术的**第三份手抄**（前两份是 `QsaShapes`、`ModelGeometry`），
而 `plan_main.cpp` 和 `device_main.cpp` **都直接传 `Geometry{}`** —— 即每个模型都按 Flash-Next 算预算。

新增 `geometry_of(core::ModelGeometry)` 转换，32K 上下文下：

| | Flash-Next | Qwen3.8-27B |
|---|---|---|
| KV + indexer | 408 MB | **1056 MB** |
| recurrent state | 112 MB | **149 MB** |

即原来的预算把一个 27B 的 KV **低估了 2.6 倍** —— 规划器返回它兑现不了的 plan，
失败落在 token 4000 而不是启动时。这正是 `DoesNotClose` 存在的意义。

> **删掉了一个自己写出来的多余东西**：一开始给 `idx_layer` 加了 `g.has_indexer() ? ... : 0` 门，
> 变异测试立刻发现是死代码 —— `geometry_of` 给出 `idx_dim = 0`，`1 * 0 / 4` 本来就是 0。
> 门等于多加一个要和宽度保持一致的东西，没有收益。

> 没改 `plan_main.cpp` / `device_main.cpp`：它们是独立探针，根本不加载模型，`Geometry{}` 是唯一可选项，
> 改名成 `flash_next()` 只是换个名字，不解决问题。真正该做的是 `generate.cpp` 走 `geometry_of(g)`。

测试 `tests/core/plan_geometry_test.cpp`，纯 CPU，14 断言。`src/plan/plan_main.cpp` 编译通过。
6 处变异全 `exit=1`。

**未完成：hc kernel 路径**

`gr.cu`(77 命中) / `fused_gr.cu`(40) / `gr_parity.cpp`(65) / `include/strata/kernels/gr.hpp`
对这两个 pack 是死代码。**本机没有 nvcc（Windows 和 WSL 都没有），编不了。**

> ponytail: 无 `hc_*` 时 `gr_read` 退化成什么？读 `gr_parity.cpp:70` 的 CPU oracle —— `hc=1` 时
> 逐流 RMSNorm + `mean` over streams 塌缩成一次普通 RMSNorm，`inject` 宽度为 1 且 `2*sigmoid` 门恒等于 1，
> 所以 `gr_write` 塌缩成 `R += block_out`。**理论上不需要新 kernel**，但 `gr_workspace_init` 要求
> DEVICE 指针且不校验、现有 elementwise.hpp 里没有 f32 拷贝原语，得先确认 aliasing 行为。

注意：`gr_*` 在 `prefill/kernels.cu` 里也占 10 处、`ple.cu` 占 45 处 —— 说明 **PLE 和 GR 在
prompt 路径里是耦合的**，删 GR 要连带看 PLE。

### 阶段 3：MoE 参数化（Qwen3.6-35B-A3B 专用，3–5 天）

Qwen3.6-35B-A3B 仍是 MoE（256 专家 / top-8 / n_ff=512），结构同族，改动相对小：

- `router_top10.cu` 的 top-10 假设要参数化（256 专家 top-8）。签名已有 `k` 参数，重点是 kernel 内部的 warp 布局
- `s2_gemv` 系列（`s2_gemv.cu` / `s2_gemv_fast.cu` / `s2_gemv_q8.cu` / `s2_expert_grouped.cu`）是 Q2_0/IQ 的
  专用 dequant+GEMV，n_ff 从 640 变 512 → 重新 tune tile
- `expert_source.cpp`(148 命中) / `expert_cache.cpp` / `peer_experts.cpp` 的 expert 寻址是
  `layer * 512 + expert` 的扁平索引 → 改成 `layer * n_expert + expert`
- `prefill/moe_fused.cu` / `moe_mmq.cu` / `moe_fused_iq.cu` 的 group 尺寸要重算
- `plan/plan.hpp` 显存预算：专家总数从 24576 降到 10240，**缓存能多装一倍专家 → 会更快**

Qwen3.8-27B **没有 MoE**，走阶段 4。

### 阶段 4：稠密 FFN（Qwen3.8-27B 专用，2–3 天）

27B 是**稠密模型**，`intermediate_size = 17408`（vs shared_expert 的 640）。
Strata 里没有稠密 FFN 的 GEMV 路径 —— `shared_expert.cu` 是最近的那个，但它的 shape 是
`[n_embd, n_ff] → [n_ff, n_embd]`，n_ff=17408 时：
- 显存：17408×5120×2 bytes = 178 MB/层 × 64 层 = **11.4 GB 常驻**，放不进 6 GB 显存，
  得走 `expert_source` 的 mmap + CPU fallback（和现在专家一个待遇）
- `intermediate_size` 是 MoE 前缀的稠密 FFN，还是全层都有？**这个必须去 model.safetensors.index.json
  里查真实的张量名**，config.json 没写 `first_k_dense_replace` / `moe_layer_freq`

### 阶段 5：量化格式（未评估，可能最大）

Strata 走 **GGUF i-quant**（从 llama.cpp 抄的 dequant），量化产物是
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` 家的 GSQ-RCO。

而 `Qwen3.6-35B-A3B-NVFP4` 你手上那个是 **NVFP4 / MXFP4**（Blackwell tensor-core 原生格式）。
这两套**没有转换路径**，需要：
- 要么用 `unsloth` 的 GGUF 重新量化目标模型（多一步，但复用 Strata 现有 dequant）
- 要么给 Strata 加 NVFP4 dequant（`src/artifact/dequant.cpp` 重写，且 `nvfp4` 需要 sm_100+，
  你的 RTX 3060 是 sm_86，**硬件就不支持**）

**建议**：一律走 GGUF 路线，别碰 NVFP4。

### 阶段 6：多模态（可选）

三个模型都有 `vision_config`。Strata 用 llama.cpp 的 `mtmd`（`tools/vision/strata_vision.cpp`）。
`out_hidden_size` 变了（5120 / 2048），`tools/embd_bf16_pack.py` 和
`src/core/native_head.cpp` 的投影维度要跟着改。

---

## 4. 硬件现实

| | 需要 | 你有（实测 nvidia-smi） |
|---|---|---|
| 显存 | ≥ 12 GB（官方最低档） | **6 GB**（RTX 3060 Laptop, 6144 MiB） |
| 内存 | ≥ 32 GB | 39.8 GB ✓ |
| 磁盘 | 66–76 GB 下载 | E 盘 389 GB ✓ |

**改造完你也跑不起来原版 Flash-Next。** 但改造完的 Qwen3.6-35B-A3B（A3B = 每 token 只激活 3B）
在 6 GB 显存 + GGUF Q4 下理论可跑 —— 前提是阶段 1–3 全做完。这才是「值得改造」的那一个。
Qwen3.8-27B 是稠密 27B，6 GB 显存跑不动，改造它主要是为了架构验证（不需要 GPU 正确性，
可以用 CPU parity 测试）。

---

## 5. 建议路线

1. **先做 Qwen3.6-35B-A3B**（MoE 同族，改动最集中，且 6 GB 显存可能真跑得动）
2. 阶段 0 + 1 + 2 + 3 全部完成后，用 `python tools/geometry_from_config.py` 的 `--check` 输出当验收清单
3. Qwen3.8-27B（稠密）等 MoE 那套稳了再上，它主要验证「没有 MoE 时 Strata 的代码路径能不能退化成稠密」
4. **跳过 NVFP4**，走 GGUF

> ponytail: 没写 `docs/` 以外的任何东西，因为现在写 C++ 就是写一堆跑不起来的 kernel。
> 先把「哪些文件要动、动的顺序、验收信号」钉死，比提前写 3000 行编译不过的 CUDA 值钱。