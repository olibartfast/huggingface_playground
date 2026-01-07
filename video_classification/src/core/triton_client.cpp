#include "video_classification/triton_client.hpp"
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include <fstream>
#include <iostream>

// Removed static ID2LABEL initialization

void TritonClient::load_labels(const std::string &labels_file) {
  if (labels_file.empty()) {
    return;
  }

  std::ifstream file(labels_file);
  if (!file.is_open()) {
    std::cerr << "Warning: Could not open labels file: " << labels_file
              << std::endl;
    return;
  }

  std::string line;
  int id = 0;
  while (std::getline(file, line)) {
    if (!line.empty()) {
      id2label_[std::to_string(id)] = line;
      id++;
    }
  }
}

TritonClient::TritonClient(const std::string &server_url,
                           const std::string &labels_file) {
  tc::Error err =
      tc::InferenceServerHttpClient::Create(&http_client_, server_url, false);
  if (!err.IsOk()) {
    throw std::runtime_error("Failed to create HTTP client: " + err.Message());
  }
  if (!labels_file.empty()) {
    load_labels(labels_file);
  }
}

// Re-adding parse_model_http functionality.
void TritonClient::parse_model_http(const rapidjson::Document &model_metadata,
                                    const rapidjson::Document &model_config,
                                    const size_t batch_size,
                                    ModelInfo *model_info) {
  const auto &input_itr = model_metadata.FindMember("inputs");
  size_t input_count =
      input_itr != model_metadata.MemberEnd() ? input_itr->value.Size() : 0;
  if (input_count != 1) {
    throw std::runtime_error("Expecting 1 input, got " +
                             std::to_string(input_count));
  }

  const auto &output_itr = model_metadata.FindMember("outputs");
  size_t output_count =
      output_itr != model_metadata.MemberEnd() ? output_itr->value.Size() : 0;
  if (output_count != 1) {
    throw std::runtime_error("Expecting 1 output, got " +
                             std::to_string(output_count));
  }

  const auto &input_config_itr = model_config.FindMember("input");
  input_count = input_config_itr != model_config.MemberEnd()
                    ? input_config_itr->value.Size()
                    : 0;
  if (input_count != 1) {
    throw std::runtime_error("Expecting 1 input in model configuration, got " +
                             std::to_string(input_count));
  }

  const auto &input_metadata = *input_itr->value.Begin();
  const auto &input_config = *input_config_itr->value.Begin();
  const auto &output_metadata = *output_itr->value.Begin();

  const auto &output_dtype_itr = output_metadata.FindMember("datatype");
  if (output_dtype_itr == output_metadata.MemberEnd()) {
    throw std::runtime_error("Output missing datatype in metadata for model '" +
                             std::string(model_metadata["name"].GetString()) +
                             "'");
  }
  std::string datatype(output_dtype_itr->value.GetString(),
                       output_dtype_itr->value.GetStringLength());
  if (datatype != "FP32") {
    throw std::runtime_error("Expecting output datatype FP32, got '" +
                             datatype + "'");
  }

  int max_batch_size = 0;
  const auto bs_itr = model_config.FindMember("max_batch_size");
  if (bs_itr != model_config.MemberEnd()) {
    max_batch_size = static_cast<int>(bs_itr->value.GetUint());
  }
  model_info->max_batch_size_ = max_batch_size;

  if (max_batch_size == 0 && batch_size != 1) {
    throw std::runtime_error("Batching not supported for model '" +
                             std::string(model_metadata["name"].GetString()) +
                             "'");
  } else if (batch_size > static_cast<size_t>(max_batch_size)) {
    throw std::runtime_error("Expecting batch size <= " +
                             std::to_string(max_batch_size));
  }

  const auto output_shape_itr = output_metadata.FindMember("shape");
  if (output_shape_itr == output_metadata.MemberEnd()) {
    throw std::runtime_error("Output missing shape in metadata for model '" +
                             std::string(model_metadata["name"].GetString()) +
                             "'");
  }
  bool output_batch_dim = (max_batch_size > 0);
  size_t non_one_cnt = 0;
  for (rapidjson::SizeType i = 0; i < output_shape_itr->value.Size(); i++) {
    if (output_batch_dim) {
      output_batch_dim = false;
    } else if (output_shape_itr->value[i].GetInt() == -1) {
      throw std::runtime_error(
          "Variable-size dimension in model output not supported");
    } else if (output_shape_itr->value[i].GetInt() > 1) {
      non_one_cnt++;
      if (non_one_cnt > 1) {
        throw std::runtime_error("Expecting model output to be a vector");
      }
    }
  }

  const bool input_batch_dim = (max_batch_size > 0);
  const size_t expected_input_dims =
      4 + (input_batch_dim ? 1 : 0); // [batch, frames, c, h, w]
  const auto input_shape_itr = input_metadata.FindMember("shape");
  if (input_shape_itr == input_metadata.MemberEnd() ||
      input_shape_itr->value.Size() != expected_input_dims) {
    throw std::runtime_error(
        "Expecting input to have " + std::to_string(expected_input_dims) +
        " dimensions, got " +
        (input_shape_itr == input_metadata.MemberEnd()
             ? "0"
             : std::to_string(input_shape_itr->value.Size())));
  }

  // Validate input shape matches [batch, 16, 3, 224, 224]
  int frame_idx = input_batch_dim ? 1 : 0;
  if (input_shape_itr->value[static_cast<rapidjson::SizeType>(frame_idx)]
              .GetInt() != 16 ||
      input_shape_itr->value[static_cast<rapidjson::SizeType>(frame_idx) + 1]
              .GetInt() != 3 ||
      input_shape_itr->value[static_cast<rapidjson::SizeType>(frame_idx) + 2]
              .GetInt() != 224 ||
      input_shape_itr->value[static_cast<rapidjson::SizeType>(frame_idx) + 3]
              .GetInt() != 224) {
    throw std::runtime_error(
        "Unexpected input shape, expecting [batch, 16, 3, 224, 224]");
  }

  // Handle input format
  model_info->input_format_ = "FORMAT_NCHW"; // Default for VideoMAE
  const auto format_itr = input_config.FindMember("format");
  if (format_itr != input_config.MemberEnd() && format_itr->value.IsString()) {
    model_info->input_format_ = std::string(
        format_itr->value.GetString(), format_itr->value.GetStringLength());
    if (model_info->input_format_ != "FORMAT_NCHW" &&
        model_info->input_format_ != "FORMAT_NHWC" &&
        model_info->input_format_ != "FORMAT_NONE") {
      throw std::runtime_error("Invalid input format '" +
                               model_info->input_format_ +
                               "', expecting FORMAT_NCHW or FORMAT_NHWC");
    }
  } else {
    std::cerr << "Warning: Input format missing in model config for '" +
                     std::string(model_metadata["name"].GetString()) +
                     "', defaulting to FORMAT_NCHW"
              << std::endl;
  }

  model_info->output_name_ =
      std::string(output_metadata["name"].GetString(),
                  output_metadata["name"].GetStringLength());
  model_info->input_name_ =
      std::string(input_metadata["name"].GetString(),
                  input_metadata["name"].GetStringLength());
  model_info->input_datatype_ =
      std::string(input_metadata["datatype"].GetString(),
                  input_metadata["datatype"].GetStringLength());

  if (model_info->input_format_ == "FORMAT_NHWC") {
    model_info->input_h_ =
        input_shape_itr->value[input_batch_dim ? 2 : 1].GetInt();
    model_info->input_w_ =
        input_shape_itr->value[input_batch_dim ? 3 : 2].GetInt();
    model_info->input_c_ =
        input_shape_itr->value[input_batch_dim ? 4 : 3].GetInt();
  } else { // FORMAT_NCHW
    model_info->input_c_ =
        input_shape_itr->value[input_batch_dim ? 2 : 1].GetInt();
    model_info->input_h_ =
        input_shape_itr->value[input_batch_dim ? 3 : 2].GetInt();
    model_info->input_w_ =
        input_shape_itr->value[input_batch_dim ? 4 : 3].GetInt();
  }

  auto parse_type = [](const std::string &dtype, int *type1, int *type3) {
    if (dtype == "FP32") {
      *type1 = CV_32FC1;
      *type3 = CV_32FC3;
      return true;
    }
    return false;
  };

  if (!parse_type(model_info->input_datatype_, &model_info->type1_,
                  &model_info->type3_)) {
    throw std::runtime_error("Unexpected input datatype '" +
                             model_info->input_datatype_ + "'");
  }
}

void TritonClient::get_model_info(const std::string &model_name,
                                  ModelInfo &model_info) {
  tc::Error err;
  std::string model_metadata;
  err = http_client_->ModelMetadata(&model_metadata, model_name,
                                    "4"); // Match version_policy
  if (!err.IsOk()) {
    throw std::runtime_error("Failed to get model metadata: " + err.Message());
  }
  rapidjson::Document model_metadata_json;
  err = tc::ParseJson(&model_metadata_json, model_metadata);
  if (!err.IsOk()) {
    throw std::runtime_error("Failed to parse model metadata: " +
                             err.Message());
  }

  std::string model_config;
  err = http_client_->ModelConfig(&model_config, model_name, "4");
  if (!err.IsOk()) {
    throw std::runtime_error("Failed to get model config: " + err.Message());
  }
  rapidjson::Document model_config_json;
  err = tc::ParseJson(&model_config_json, model_config);
  if (!err.IsOk()) {
    throw std::runtime_error("Failed to parse model config: " + err.Message());
  }

  parse_model_http(model_metadata_json, model_config_json, 1, &model_info);
}

std::vector<TritonClient::InferenceResult>
TritonClient::infer(const std::vector<float> &input_data,
                    const std::string &model_name, const ModelInfo &model_info,
                    const std::vector<int64_t> &shape) {
  tc::Error err;

  tc::InferInput *input;
  err = tc::InferInput::Create(&input, model_info.input_name_, shape,
                               model_info.input_datatype_);
  if (!err.IsOk()) {
    throw std::runtime_error("Failed to create input: " + err.Message());
  }
  std::shared_ptr<tc::InferInput> input_ptr(input);

  err =
      input_ptr->AppendRaw(reinterpret_cast<const uint8_t *>(input_data.data()),
                           input_data.size() * sizeof(float));
  if (!err.IsOk()) {
    throw std::runtime_error("Failed to set input data: " + err.Message());
  }

  tc::InferRequestedOutput *output;
  err = tc::InferRequestedOutput::Create(&output, model_info.output_name_);
  if (!err.IsOk()) {
    throw std::runtime_error("Failed to create output: " + err.Message());
  }
  std::shared_ptr<tc::InferRequestedOutput> output_ptr(output);

  std::vector<tc::InferInput *> inputs = {input_ptr.get()};
  std::vector<const tc::InferRequestedOutput *> outputs = {output_ptr.get()};

  tc::InferOptions options(model_name);
  options.model_version_ = "4"; // Match version_policy

  tc::InferResult *result;
  err = http_client_->Infer(&result, options, inputs, outputs);
  if (!err.IsOk()) {
    throw std::runtime_error("Inference failed: " + err.Message());
  }
  std::unique_ptr<tc::InferResult> result_ptr(result);

  std::vector<float> logits;
  const float *output_data;
  size_t output_size;
  err = result_ptr->RawData(model_info.output_name_,
                            reinterpret_cast<const uint8_t **>(&output_data),
                            &output_size);
  if (!err.IsOk()) {
    throw std::runtime_error("Failed to get output data: " + err.Message());
  }
  logits.assign(output_data, output_data + output_size / sizeof(float));

  return postprocess_results(logits);
}

std::vector<TritonClient::InferenceResult>
TritonClient::postprocess_results(const std::vector<float> &logits) {
  std::vector<InferenceResult> results;

  std::vector<float> probs(logits.size());
  float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = std::exp(logits[i] - max_logit);
    sum += probs[i];
  }
  for (auto &p : probs) {
    p /= sum;
  }

  std::vector<int> indices(probs.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&probs](int a, int b) {
    return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)];
  });

  for (int i = 0; i < std::min(3, static_cast<int>(indices.size())); ++i) {
    int idx = indices[static_cast<size_t>(i)];
    InferenceResult result;
    if (id2label_.count(std::to_string(idx))) {
      result.label = id2label_.at(std::to_string(idx));
    } else {
      result.label = "unknown_" + std::to_string(idx);
    }
    result.probability = probs[static_cast<size_t>(idx)];
    results.push_back(result);
  }

  return results;
}
