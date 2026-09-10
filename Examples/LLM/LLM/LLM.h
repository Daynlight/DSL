// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/chrono.h>
#include <fmt/std.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>
#include <poll.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <string_view>
#include <cstdint>
#include <limits>
#include <chrono>
#include <charconv>

#include <NeuralNetwork/NeuralNetwork.h>
#include <NeuralNetwork/Utils.h>



namespace LLM{
template<size_t V, size_t E>
class Embedding{

private:
  std::vector<float> data;

public:
  Embedding() noexcept;
  ~Embedding() noexcept;

  void zero(size_t token) noexcept;

  template<size_t C>
  void forward(const std::array<size_t, C>& tokens, std::array<float, C * E>& output) const noexcept;

  template<size_t C>
  void backprop(const std::array<size_t, C>& tokens, const float* gradient, float learning_rate, size_t frozen_token = V) noexcept;

  std::string serialize() const noexcept;
  bool deserialize(std::string_view input) noexcept;

  float* getData() noexcept;
  const float* getData() const noexcept;
  size_t size() const noexcept;

};



class LLM{
// ======================================== //
// ================= Data ================= //
// ======================================== //
private:
  std::filesystem::path path_to_learning_set = std::filesystem::path(__FILE__).parent_path() / "../data/learn_set.txt";
  std::filesystem::path path_to_model_data = std::filesystem::path(__FILE__).parent_path() / "../data/model";
  bool model_is_loaded_from_file = false;
  bool is_running = true;

  float learning_rate = 0.001;
  unsigned int learn_samples = 1000;
  unsigned int epoch = 200;
  unsigned int learning_set_repeats = 1;
  unsigned int max_lr_reductions = 5;
  bool additional_acuracy_show = true;
  bool balanced_learning = false;
  bool dynamic_lr = false;
  bool gpu_acceleration = true;

  static constexpr size_t context_size = 16;
  static constexpr size_t embedding_size = 1024;
  static constexpr size_t input_size = context_size * embedding_size;
  static constexpr size_t vocab_size = 8192;
  static constexpr size_t response_size = 256;

  Embedding<vocab_size, embedding_size> embedding;
  static constexpr std::array<size_t, 9> model_shape = {input_size, 16384, 8192, 4096, 4096, 2048, 1024, vocab_size, 1};
  NN::NeuralNetwork<input_size, 16384, 8192, 4096, 4096, 2048, 1024, vocab_size, 1> model;

  std::filesystem::path path_to_tokens = std::filesystem::path(__FILE__).parent_path() / "../data/tokens";

  std::vector<std::string> tokens;
  std::unordered_map<std::string, size_t> token_to_id;

  static constexpr size_t unk_token = 0;
  static constexpr size_t pad_token = 1;

  std::filesystem::path path_to_info = std::filesystem::path(__FILE__).parent_path() / "../data/info";
  unsigned int update_counter = 0;
  unsigned int request_counter = 0;
  float last_ce = 0.0f;
  float best_ce = std::numeric_limits<float>::max();
  float last_mse = 0.0f;
  float best_mse = std::numeric_limits<float>::max();
  float last_accuracy = 0.0f;
  float best_accuracy = 0.0f;

  double last_update_time_ms = 0.0;
  double total_update_time_ms = 0.0;

  std::filesystem::file_time_type info_last_write{};


// ======================================== //
// ============== Functions =============== //
// ======================================== //
// =========================== //
// ======= Constructors ====== //
// =========================== //
public:
  LLM() noexcept;
  ~LLM() noexcept;

// =========================== //
// ========= Control ========= //
// =========================== //
public:
  void setLearnFilePath(std::filesystem::path path) noexcept;
  std::filesystem::path getLearnFilePath() const noexcept;
  void setLearningRate(float value) noexcept;
  void setLearningEpoch(int value) noexcept;
  void setLearningSamples(int value) noexcept;
  void setLearningSetRepeats(int value) noexcept;
  void setBalancedLearning(int value) noexcept;
  void setGpuAcceleration(int value) noexcept;
  void setDynamicLearningRate(int value) noexcept;
  void setAdditionalAcuracyShow(int value) noexcept;
  void setMaxLearningRateReductions(int value) noexcept;
  void setModelFilePath(std::filesystem::path path) noexcept;
  std::filesystem::path getModelFilePath() const noexcept;
  void updateInfo() noexcept;
  
// =========================== //
// ====== Serialization ====== //
// =========================== //
public:
  void loadModelFromFile() noexcept;
  void saveModelToFile() noexcept;
  void saveEmbeddingToFile() const noexcept;
  bool loadEmbeddingFromFile() noexcept;
  void saveInfoToFile() noexcept;
  void loadInfoFromFile() noexcept;

// =========================== //
// ========= Helpers ========= //
// =========================== //
public:
  std::string readLearningSet() const noexcept;
  std::string prepareLearningSet(const std::string& text) const noexcept;
  template<size_t S>
  std::pair<float, std::array<float, S>> calculateLearningSetEntropy(const std::string& text) const noexcept;
  void setModel() noexcept;
  void learningInfo(bool show_learning_ifno, const std::array<float, vocab_size>& target, size_t expected, float& ce_sum, float& mse_sum, size_t& correct, size_t& log_samples) noexcept;

  void createTokens(const std::string& text) noexcept;
  void saveTokensToFile() const noexcept;
  bool loadTokensFromFile() noexcept;

  std::vector<size_t> encodeTokens(const std::string& text) const noexcept;
  std::string decodeTokens(const std::vector<size_t>& ids) const noexcept;

// =========================== //
// ======= Application ======= //
// =========================== //
public:
  void learn() noexcept;
  std::string getRespond(const std::string& message) noexcept;
  
  void printHelp() const noexcept;

  void onStart() noexcept;
  void onUpdate() noexcept;
  void onEnd() noexcept;
  void application() noexcept;

};
};



#include "LLM.hpp"