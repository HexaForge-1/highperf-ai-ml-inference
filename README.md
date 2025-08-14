# 🚀 highperf-ai-ml-inference (Unified CPU+GPU)
High-Performance C++ AI/ML inference (CLI + REST) with **ONNX Runtime** (CPU default, GPU opt‑in) and optional **LibTorch**. Cross‑platform via CMake/Docker/Codespaces. Includes asset scripts, ImageNet labels, matrix CI and PR merge gating.

## CPU Quickstart (Linux / Codespaces)
```bash
bash scripts/fetch_assets.sh
cmake -S . -B build -DBUILD_ONNXRUNTIME=ON -DBUILD_REST_API=ON
cmake --build build --config Release -j
./build/bin/highperf-ai-ml-inference --backend onnx --model models/squeezenet1.1.onnx --input assets/sample.jpg --topk 5
ctest --test-dir build --output-on-failure
```

## GPU Quickstart (Linux with NVIDIA)
```bash
# Fetch CUDA ONNX Runtime
ORT_USE_CUDA=1 bash scripts/fetch_assets.sh

# Build (auto-links libonnxruntime_gpu.so if present)
cmake -S . -B build -DBUILD_ONNXRUNTIME=ON -DBUILD_REST_API=ON
cmake --build build --config Release -j

# Run
./build/bin/highperf-ai-ml-inference --backend onnx --model models/squeezenet1.1.onnx --input assets/sample.jpg --topk 5
```
> Use `-DFORCE_CPU_ORT=ON` in CMake if you fetched GPU libs but want to **force CPU** link.

### Runtime safeguard
If the binary is linked against the GPU ORT but **CUDA runtime** isn’t available, you’ll see:
```
[warn] ONNX Runtime GPU library linked but CUDA runtime not detected...
```
This is a friendly hint to either install drivers or rebuild for CPU.

## REST API
```bash
./build/bin/highperf-ai-ml-inference --backend onnx --model models/squeezenet1.1.onnx --serve 8080
curl -X POST "http://localhost:8080/predict?file=assets/sample.jpg"
```

## Optional: LibTorch (TorchScript)
```bash
# CPU
bash scripts/fetch_libtorch.sh
cmake -S . -B build -DBUILD_ONNXRUNTIME=OFF -DBUILD_LIBTORCH=ON
cmake --build build --config Release -j

# CUDA (requires NVIDIA + CUDA)
LIBTORCH_CUDA=1 bash scripts/fetch_libtorch.sh
cmake -S . -B build -DBUILD_ONNXRUNTIME=OFF -DBUILD_LIBTORCH=ON
cmake --build build --config Release -j
```

## CI
PRs to `main` are gated by **CPU** matrix CI (Ubuntu + Windows).

## License
MIT
