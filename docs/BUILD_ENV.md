# 本机构建环境

结论：**WSL + apt 装的 CUDA 12.9**。已实测 nvcc 编出 sm_86 cubin 并在 RTX 3060 Laptop 上跑出正确结果。

## 为什么不是 Docker

- 项目只发 Windows 预编译引擎（`strata-windows-x64-cuda12.zip`，v0.1.39）。
  `nvidia/cuda` 是 Linux 镜像，编出的 `.so` 是 Linux ELF，在这台机器上跑不了 —— 要 Docker 就得整台搬到 Linux。
- 本机 Docker daemon 当时也没跑（`npipe:////./pipe/dockerDesktopLinuxEngine` 不存在）。

## 为什么不是 venv

venv 隔离 Python。`nvcc` 是 C++/CUDA 编译器，与 Python 无关。venv 在这道题上帮不上任何忙。

## 为什么选 12.9 而不是 13.x

`setup.py:2150` 的门槛是 `need_cuda = (13,0) if max(archs) >= 120 else (12,0)`。
RTX 3060 Laptop 是 sm_86 → 只要 **12.0**。13.x 只有 sm_120（RTX 50）才需要。

驱动 615.78.02，CUDA 12.9 要求 525，够。

## 装了什么（12 个包，没有驱动没有 dkms）

```
cuda-nvcc-12-9 cuda-cudart-dev-12-9 libcublas-dev-12-9 cuda-cccl-12-9
  -> cuda-crt-12-9 cuda-cudart-12-9 cuda-driver-dev-12-9 cuda-nvvm-12-9
     cuda-toolkit-12-9-config-common cuda-toolkit-12-config-common
     cuda-toolkit-config-common libcublas-12-9
```

```bash
# keyring 装一次，之后 apt 自己管；不用 trusted=yes 绕签名
cd /tmp && curl -sSLO https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y --no-install-recommends \\
  cuda-nvcc-12-9 cuda-cudart-dev-12-9 libcublas-dev-12-9 cuda-cccl-12-9 cmake ninja-build
```

`--no-install-recommends` 是必须的：默认会拖进 `dkms` + `nvidia-installer-cleanup`，
而 WSL 的内核模块由 Windows 驱动提供，dkms 装不了也没用。

Debian 官方源只有 `nvidia-cuda-toolkit` 11.8 —— **不够**，必须走 NVIDIA 自己的 repo。

### cmake 要换掉 apt 的

apt 的 cmake 3.25 **不认** nvcc 12.9 的 `CUDA20` dialect，configure 直接报
`requires the language dialect "CUDA20"`（7 个 target 全挂）。

用 Kitware 官方二进制（一个 tar，无依赖，55 MB）：

```bash
curl -sSL -o /tmp/cmake.tar.gz \\
  https://gh-proxy.com/https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz
tar -xzf /tmp/cmake.tar.gz -C /opt && mv /opt/cmake-3.31.6-linux-x86_64 /opt/cmake
```

项目自己只要 3.24（`cmake_minimum_required`），但要认 nvcc 的 C++20 dialect 得 3.28+。


## 用的时候

```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH
```

WSL 继承 Windows 的 PATH（含空格和括号），任何内联 `bash -c '...'` 都会 syntax error。
写脚本文件跑，别内联。

源码路径：`/opt/src/strata` → `/mnt/e/zip/agent file big/01_项目代码/Strata`（ASCII symlink，绕开中文路径）。

## 现状

- nvcc 12.9.86 ✓ / cmake 3.25.1 ✓（要求 3.24）/ ninja 1.11.1 ✓ / g++ 12.2.0 ✓
- `third_party/ggml` 空 —— llama.cpp 需按 `setup.py` 的 `LLAMA_CPP_COMMIT` 拉（GitHub 直连通，39.5 MB）
- 完整 CMake 构建未跑


## 从 Windows 主机驱动 WSL（本机实测的坑）

从 git-bash 或任何 Windows 进程调 `wsl.exe` 时，命令行会先过一遍 Windows 的解析：

- **shell 变量会被吃空。** `wsl.exe -- bash -c 'X=1; echo $X'` 打印空 — `$X` 在到达
  WSL 前就被当成了 Windows 环境变量。**写死完整路径**，不要在跨边界的命令里用 `$VAR`。
  同理 `sed -n '1,5p;7,9p'` 的分号和 `grep 'foo('` 的括号都会被吃掉。
- **中文目录名会损坏。** `mkdir -p "模型"` 在跨边界时建成乱码目录（实测 `妯″瀷`），
  curl 会照着那个乱码路径写。**长任务用 ASCII 路径**；已经建错了就 `shutil.move` 到正确位置
  （curl 的 `-C -` 可以从断点续传，不会浪费已下的部分）。
- **`subprocess` 传 `input=` 给 git 会挂死。** `git commit` 等 stdin 时若 stdin 是管道，
  进程等的是 EOF 而不是你的文本。**提交信息写文件，用 `git commit -F <file>`。**
- **单次命令有 300 秒上限。** ctest 全量、`cmake --build` 这类长任务要放到后台跑并落盘日志，
  不要在前台等。

已经踩过的具体现场：Qwen3.6-35B-A3B 的 IQ2_XXS（9.94 GB）从 hf-mirror 下载时，
目录名乱码导致文件落在错误路径，中途还要 `mv` 回正确位置再续传。
