"""Copy a real Qwen3.6 GGUF's header into a small standalone file, so tests/core/q35_real_header_test.cpp can
check read_geometry against the REAL metadata without the 9.94 GB download.

Only the magic, the KV block and the tensor descriptors are copied - no tensor DATA, and nothing reads it:
read_geometry looks at metadata and shapes only.  The bytes come from the model itself, not from gguf-python and
numbers chosen to make a test pass.  ~11 MB out of 9.94 GB, almost all of it tokenizer.ggml.tokens.

  python3 tools/copy_gguf_header.py <big.gguf> <small.gguf>
"""
import os
import struct
import sys

ALIGN = 32
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        if f.read(4) != b"GGUF":
            print("%s is not a GGUF" % src)
            return 1
        f.seek(0)
        f.read(4)                                              # magic, already checked above
        version, = struct.unpack("<I", f.read(4))
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))

        def gstr():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n)

        for _ in range(n_kv):
            gstr()
            t, = struct.unpack("<I", f.read(4))
            if t == 8:                                            # STRING
                n, = struct.unpack("<Q", f.read(8))
                f.seek(n, 1)
            elif t == 9:                                          # ARRAY: elem type, then count
                et, = struct.unpack("<I", f.read(4))
                n, = struct.unpack("<Q", f.read(8))
                if et == 8:
                    for _ in range(n):
                        m, = struct.unpack("<Q", f.read(8))
                        f.seek(m, 1)
                else:
                    f.seek(SZ[et] * n, 1)
            else:
                f.seek(SZ[t], 1)
        kv_end = f.tell()

        for _ in range(n_tensors):
            gstr()
            nd, = struct.unpack("<I", f.read(4))
            f.seek(8 * nd, 1)
            f.seek(4 + 8, 1)                                      # ggml type + offset
        ti_end = f.tell()

        f.seek(0)
        out = f.read(ti_end)

    out += b"\0" * ((-len(out)) % ALIGN)                          # the data section starts 32-byte aligned
    with open(dst, "wb") as o:
        o.write(out)
    print("%s: %d tensors, %d KiB (header of %s)" % (dst, n_tensors, os.path.getsize(dst) >> 10, src))
    return 0


raise SystemExit(main())
