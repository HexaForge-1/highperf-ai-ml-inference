#pragma once
#include <string>
#include <vector>

struct InferResult {
  std::vector<int>   indices;
  std::vector<float> scores;
};

class InferenceEngine {
public:
  bool init(const std::string& backend, const std::string& model_path, int threads);
  InferResult classify(const unsigned char* rgb, int w, int h, int c, int topk);

private:
  enum class Backend { ONNX, TORCH } backend_ = Backend::ONNX;
  void* impl_ = nullptr; // ONNXBackend* or TorchBackend*
};
