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

#include <NeuralNetwork/NeuralNetwork.h>
#include <NeuralNetwork/Utils.h>



namespace LLM{
class LLM{
// ======================================== //
// ================= Data ================= //
// ======================================== //
private:
  std::filesystem::path path_to_learning_set = std::filesystem::path(__FILE__).parent_path() / "../data/learn_set.txt";
  std::filesystem::path path_to_model_data = std::filesystem::path(__FILE__).parent_path() / "../data/model";
  bool model_is_loaded_from_file = false;
  bool is_running = true;

  double learning_rate = 0.001;
  unsigned int learn_samples = 1000;
  unsigned int epoch = 200;
  unsigned int learning_set_repeats = 1;
  unsigned int max_lr_reductions = 5;
  bool additional_acuracy_show = true;
  bool balanced_learning = false;
  bool dynamic_lr = false;

  static constexpr size_t context_size = 16;
  static constexpr size_t vocab_size = 1000;
  static constexpr size_t input_size = context_size * vocab_size;
  static constexpr size_t response_size = 256;

  NN::NeuralNetwork<input_size, 8192, 4096, 2048, 1024, vocab_size, 1> model;

  std::filesystem::path path_to_tokens = std::filesystem::path(__FILE__).parent_path() / "../data/tokens";

  std::vector<std::string> tokens;
  std::unordered_map<std::string, size_t> token_to_id;

  static constexpr size_t unk_token = 0;
  static constexpr size_t pad_token = 1;



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
  void setLearningRate(double value) noexcept;
  void setLearningEpoch(int value) noexcept;
  void setLearningSamples(int value) noexcept;
  void setLearningSetRepeats(int value) noexcept;
  void setBalancedLearning(int value) noexcept;
  void setDynamicLearningRate(int value) noexcept;
  void setAdditionalAcuracyShow(int value) noexcept;
  void setModelFilePath(std::filesystem::path path) noexcept;
  std::filesystem::path getModelFilePath() const noexcept;
  
// =========================== //
// ====== Serialization ====== //
// =========================== //
public:
  void loadModelFromFile() noexcept;
  void saveModelToFile() noexcept;

// =========================== //
// ========= Helpers ========= //
// =========================== //
public:
  std::string readLearningSet() const noexcept;
  std::string prepareLearningSet(const std::string& text) const noexcept;
  template<size_t S>
  std::pair<double, std::array<double, S>> calculateLearningSetEntropy(const std::string& text) const noexcept;
  void setModel() noexcept;
  void learningInfo(bool show_learning_ifno, const std::array<double, vocab_size>& target, size_t expected, double& ce_sum, double& mse_sum, size_t& correct, size_t& log_samples) noexcept;

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