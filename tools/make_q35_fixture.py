#!/usr/bin/env python3
"""Metadata-only qwen35moe GGUFs with Qwen3.6-35B-A3B's real geometry (2 layers so they build in a second).

Why this exists: the other geometry test hand-writes GGUF bytes, so it only proves the reader agrees with
our own writer.  These are written by gguf-python, so read_geometry is checked against a real encoder.
No tensors are needed - read_geometry reads metadata - so the files stay under a kilobyte.

  python3 make_q35_fixture.py out.gguf          # carries expert_feed_forward_length
  python3 make_q35_fixture.py out.gguf --no-routed-ff   # omits it, so the /expert_used_count branch runs

The second variant exists because read_geometry takes the routed width from expert_feed_forward_length when
the file has it, and only falls back to `shared / expert_used_count` when it does not.  A file WITH the key
never reaches that division, so a test on it alone cannot tell the division from its absence.
"""
import pathlib
import sys

import gguf

# Qwen3.6-35B-A3B, except block_count is 2 to keep the file instant.  Every other number is the real one,
# and they are chosen to break the Flash-Next defaults: 2048 != 2560, 256 experts != 512, 64 != 640.
N_EMBD, N_FF, NE, LAYERS, TOP_K = 2048, 64, 256, 2, 8

w = gguf.GGUFWriter(sys.argv[1], "qwen35moe")
w.add_uint32("qwen35moe.embedding_length", N_EMBD)
w.add_uint32("qwen35moe.block_count", LAYERS)
w.add_uint32("qwen35moe.attention.head_count", 16)
w.add_uint32("qwen35moe.attention.head_count_kv", 2)
w.add_uint32("qwen35moe.expert_count", NE)
w.add_uint32("qwen35moe.expert_used_count", TOP_K)
if "--no-routed-ff" not in sys.argv:
    w.add_uint32("qwen35moe.expert_feed_forward_length", N_FF)
w.add_uint32("qwen35moe.expert_shared_feed_forward_length", 512)
w.add_uint32("qwen35moe.full_attention_interval", 4)
w.add_uint32("qwen35moe.ssm.state_size", 128)
w.add_uint32("qwen35moe.ssm.conv_kernel", 4)
# llama-arch.cpp's spelling: ssm.group_count is the KEY heads, ssm.time_step_rank the VALUE heads
w.add_uint32("qwen35moe.ssm.group_count", 16)
w.add_uint32("qwen35moe.ssm.time_step_rank", 32)
w.add_uint32("qwen35moe.ssm.value_dim", 64)
# no attention.key_length on purpose: head_dim must be derived n_embd/n_head, not defaulted to 256
# no tensors: read_geometry reads metadata only
w.write_header_to_file()
w.write_kv_data_to_file()
w.close()
# gguf-python writes no padding, but Strata's reader aligns the data section to 32 bytes and refuses a file
# whose header ends past it.  With zero tensors there is no data section, so pad the file to the alignment.
p = pathlib.Path(sys.argv[1])
raw = p.read_bytes()
align = 32
pad = (-len(raw)) % align
if pad:
    p.write_bytes(raw + b"\0" * pad)
routed = N_FF if "--no-routed-ff" not in sys.argv else "512/%d derived" % TOP_K
print(f"wrote {sys.argv[1]}: {N_EMBD} embd, {LAYERS} layers, {NE} experts top-{TOP_K}, routed ff {routed}")
