#pragma once

#include "json_utils.hpp"
#include <http_client.h>
#include <map>
#include <memory>
#include <rapidjson/document.h>
#include <string>
#include <vector>

namespace tc = triton::client; // Namespace alias

struct ModelInfo {
  std::string output_name_;
  std::string input_name_;
  std::string input_datatype_;
  int input_c_;
  int input_h_;
  int input_w_;
  std::string input_format_;
  int type1_;
  int type3_;
  int max_batch_size_;
};

class TritonClient {
public:
  struct InferenceResult {
    std::string label;
    float probability;
  };

  TritonClient(const std::string &server_url,
               const std::string &labels_file = "");
  std::vector<InferenceResult> infer(const std::vector<float> &input_data,
                                     const std::string &model_name,
                                     const ModelInfo &model_info,
                                     const std::vector<int64_t> &shape);
  void get_model_info(const std::string &model_name, ModelInfo &model_info);

private:
  std::vector<InferenceResult>
  postprocess_results(const std::vector<float> &logits);
  static void parse_model_http(const rapidjson::Document &model_metadata,
                               const rapidjson::Document &model_config,
                               const size_t batch_size, ModelInfo *model_info);
  void load_labels(const std::string &labels_file);

  std::unique_ptr<tc::InferenceServerHttpClient> http_client_;
  std::map<std::string, std::string> id2label_;
};
