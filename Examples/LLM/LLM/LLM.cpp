// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "LLM.h"



bool hasInput() noexcept {
  pollfd fd{
    .fd = STDIN_FILENO,
    .events = POLLIN,
    .revents = 0
  };

  return poll(&fd, 1, 0) > 0 && (fd.revents & POLLIN);
};



// =========================== //
// ======= Constructors ====== //
// =========================== //
// core
LLM::LLM::LLM() noexcept { };



LLM::LLM::~LLM() noexcept { };



// =========================== //
// ========= Control ========= //
// =========================== //
void LLM::LLM::setLearnFilePath(std::filesystem::path path) noexcept {
  if(!std::filesystem::exists(std::filesystem::path(__FILE__).parent_path() / path) || 
      std::filesystem::is_directory(std::filesystem::path(__FILE__).parent_path() / path)) return;
  path_to_learning_set = std::filesystem::path(__FILE__).parent_path() / path;
};



std::filesystem::path LLM::LLM::getLearnFilePath() const noexcept {
  return path_to_learning_set;
};



void LLM::LLM::setLearningRate(float value) noexcept {
  if(value < 0.00000001) return;
  learning_rate = value;
};



void LLM::LLM::setLearningEpoch(int value) noexcept {
  if(value < 0) return;
  epoch = value;
};



void LLM::LLM::setLearningSamples(int value) noexcept {
  if(value < 0) return;
  learn_samples = value;
};



void LLM::LLM::setLearningSetRepeats(int value) noexcept {
  if(value < 0) return;
  learning_set_repeats = value;
};



void LLM::LLM::setBalancedLearning(int value) noexcept {
  if(value <= 0) balanced_learning = false;
  else balanced_learning = true;
};



void LLM::LLM::setDynamicLearningRate(int value) noexcept {
  if(value <= 0) dynamic_lr = false;
  else dynamic_lr = true;
};



void LLM::LLM::setAdditionalAcuracyShow(int value) noexcept {
  if(value <= 0) additional_acuracy_show = false;
  else additional_acuracy_show = true;
};



void LLM::LLM::setModelFilePath(std::filesystem::path path) noexcept {
  if(!std::filesystem::exists(std::filesystem::path(__FILE__).parent_path() / path) || 
      std::filesystem::is_directory(std::filesystem::path(__FILE__).parent_path() / path)) return;
  path_to_model_data = std::filesystem::path(__FILE__).parent_path() / path;
};



std::filesystem::path LLM::LLM::getModelFilePath() const noexcept {
  return path_to_model_data;
};



// =========================== //
// ====== Serialization ====== //
// =========================== //
void LLM::LLM::loadModelFromFile() noexcept {
  if(!std::filesystem::exists(path_to_model_data) || std::filesystem::is_directory(path_to_model_data)){
    model.setWeights<0>(-0.00968, 0.00968);
    model.setWeights<1>(-0.02706, 0.02706);
    model.setWeights<2>(-0.02706, 0.02706);
    model.setWeights<3>(-0.06275, 0.06275);
    model.setWeights<4>(-0.06275, 0.06275);
    model.setWeights<5>(-0.03827, 0.03827);
    return;
  };

  std::ifstream file(path_to_model_data, std::ios::binary);
  if(!file.is_open()) return;

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string data(size, '\0');
  if(size > 0) file.read(data.data(), size);
  if(!file) return;

  model.deserialize(data);
};



void LLM::LLM::saveModelToFile() noexcept {
  std::ofstream file(path_to_model_data, std::ios::binary);
  if(!file.is_open()) return;
  std::string data = model.serialize();

  if(!data.empty()) file.write(data.data(), data.size());
};



// =========================== //
// ========= Helpers ========= //
// =========================== //
std::string LLM::LLM::readLearningSet() const noexcept {
  if(!std::filesystem::exists(path_to_learning_set) || std::filesystem::is_directory(path_to_learning_set)) return "";

  std::ifstream file(path_to_learning_set, std::ios::binary);
  if(!file.is_open()) return "";
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  if(file_size <= context_size) return "";

  std::string text(file_size, '\0');
  file.read(text.data(), file_size);
  if(!file) return "";

  return text;
};



std::string LLM::LLM::prepareLearningSet(const std::string& text) const noexcept {
  std::stringstream stream(text);

  std::string line;
  std::string compact_text;

  while(std::getline(stream, line)){
    if(line.empty()) continue;
    while(!line.empty() && std::isspace(static_cast<unsigned char>(line.back())))
      line.pop_back();
    if(line.empty()) continue;
    if(line.back() == ':') continue;
    if(!compact_text.empty())
      compact_text += ' ';
    compact_text += line;
  };

  std::string normalized_text;
  normalized_text.reserve(compact_text.size());
  bool last_was_space = false;

  for(unsigned char character : compact_text){
    if(std::isspace(character)){
      if(!last_was_space)
        normalized_text += ' ';
      last_was_space = true;
    }else{
      normalized_text += static_cast<char>(character);
      last_was_space = false;
    };
  };

  return normalized_text;
};



template<size_t S>
std::pair<float, std::array<float, S>> LLM::LLM::calculateLearningSetEntropy(const std::string &text) const noexcept {
  std::array<float, S> frequency{};
  size_t total = 0;

  for(unsigned char c : text){
    if(c >= S) c = '?';
    frequency[c]++;
    total++;
  };

  float entropy = 0.0;
  for(size_t i = 0; i < S; i++){
    if(frequency[i] == 0) continue;
    frequency[i] = static_cast<float>(frequency[i]) / static_cast<float>(total);
    entropy -= frequency[i] * std::log(frequency[i]);
  };

  return {entropy, frequency};
};



void LLM::LLM::setModel() noexcept {
  model.setLearningRate(learning_rate);
  model.setActivation<0, NN::Linear>();
  model.setActivation<1, NN::ReLU>();
  model.setActivation<2, NN::ReLU>();
  model.setActivation<3, NN::ReLU>();
  model.setActivation<4, NN::ReLU>();
  model.setActivation<5, NN::Softmax>();
  model.setLoss<5, NN::CrossEntropy>();
  // model.setGPUAcceleration(false);
};



void LLM::LLM::learningInfo(bool show_learning_ifno, const std::array<float, vocab_size>& target, size_t expected, float &ce_sum, float &mse_sum, size_t &correct, size_t &log_samples) noexcept {
  float* result = model.getActivatedResult();

  size_t predicted = 0;

  for(size_t k = 1; k < vocab_size; k++) if(result[k] > result[predicted]) predicted = k;
  
  float mse = 0.0;
  for(size_t k = 0; k < vocab_size; k++){
    float diff = result[k] - target[k];
    mse += diff * diff / 2.0;
  };
  mse /= vocab_size;

  float probability = result[expected];
  float ce = -std::log(std::max(probability, 0.0000001f));

  mse_sum += mse;
  ce_sum += ce;
  correct += predicted == expected;
  log_samples++;

  if(show_learning_ifno){
    fmt::println("");
    fmt::println(
      "Average MSE: {}, CE: {}, Accuracy: {}%",
      mse_sum / log_samples,
      ce_sum / log_samples,
      100.0 * correct / log_samples
    );

    mse_sum = 0.0;
    ce_sum = 0.0;
    correct = 0;
    log_samples = 0;
  };
};



void LLM::LLM::createTokens(const std::string& text) noexcept {
  tokens.clear();
  token_to_id.clear();

  tokens.emplace_back("<UNK>");
  tokens.emplace_back("<PAD>");

  token_to_id.emplace("<UNK>", unk_token);
  token_to_id.emplace("<PAD>", pad_token);

  std::unordered_map<std::string, size_t> frequency;

  std::istringstream stream(text);
  std::string word;

  while(stream >> word) frequency[word]++;

  std::vector<std::pair<std::string, size_t>> sorted;
  sorted.reserve(frequency.size());

  for(const auto& [word, count] : frequency) sorted.emplace_back(word, count);
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b){
    if(a.second != b.second) return a.second > b.second;
    return a.first < b.first;
  });

  for(const auto& [word, count] : sorted){
    if(tokens.size() >= vocab_size) break;
    if(word == "<UNK>" || word == "<PAD>") continue;

    size_t id = tokens.size();
    token_to_id.emplace(word, id);
    tokens.push_back(word);
  };

  while(tokens.size() < vocab_size){
    std::string token = "<UNUSED_" + std::to_string(tokens.size()) + ">";
    token_to_id.emplace(token, tokens.size());
    tokens.push_back(std::move(token));
  };
};



void LLM::LLM::saveTokensToFile() const noexcept {
  std::ofstream file(path_to_tokens, std::ios::binary | std::ios::trunc);
  if(!file.is_open()) return;

  uint64_t count = tokens.size();
  file.write(reinterpret_cast<const char*>(&count), sizeof(count));

  for(const std::string& token : tokens){
    uint64_t size = token.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(token.data(), size);
  };
};



bool LLM::LLM::loadTokensFromFile() noexcept {
  std::ifstream file(path_to_tokens, std::ios::binary);
  if(!file.is_open()) return false;

  uint64_t count = 0;
  file.read(reinterpret_cast<char*>(&count), sizeof(count));
  if(!file || count != vocab_size) return false;

  std::vector<std::string> loaded_tokens;
  std::unordered_map<std::string, size_t> loaded_ids;

  for(uint64_t i = 0; i < count; i++){
    uint64_t size = 0;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    if(!file || size > 4096) return false;

    std::string token(size, '\0');

    if(size > 0) file.read(token.data(), size);
    if(!file) return false;
    if(loaded_ids.contains(token)) return false;

    loaded_ids.emplace(token, loaded_tokens.size());
    loaded_tokens.push_back(std::move(token));
  };

  if(loaded_tokens[unk_token] != "<UNK>" || loaded_tokens[pad_token] != "<PAD>") return false;

  tokens = std::move(loaded_tokens);
  token_to_id = std::move(loaded_ids);

  return true;
};



std::vector<size_t> LLM::LLM::encodeTokens(const std::string& text) const noexcept {
  std::vector<size_t> result;
  std::istringstream stream(text);
  std::string word;

  while(stream >> word){
    auto it = token_to_id.find(word);

    if(it == token_to_id.end()) result.push_back(unk_token);
    else result.push_back(it->second);
  };

  return result;
};



std::string LLM::LLM::decodeTokens(const std::vector<size_t>& ids) const noexcept {
  std::string result;

  for(size_t id : ids){
    if(id >= tokens.size()) continue;
    if(id == pad_token) continue;
    if(!result.empty()) result += ' ';
    result += tokens[id];
  };

  return result;
};



// =========================== //
// ======= Application ======= //
// =========================== //
void LLM::LLM::learn() noexcept {
  if(!std::filesystem::exists(path_to_learning_set) || std::filesystem::is_directory(path_to_learning_set)) return;

  fmt::println(fg(fmt::color::yellow), "-- Parameters");
  fmt::println(fg(fmt::color::blue), "Path To Learning Set: {}", path_to_learning_set);
  fmt::println(fg(fmt::color::blue), "Path To Model Data: {}", path_to_model_data);
  fmt::println(fg(fmt::color::blue), "Path To Tokens: {}", path_to_tokens);
  fmt::println(fg(fmt::color::blue), "Context Size: {}", context_size);
  fmt::println(fg(fmt::color::blue), "Vocab Size: {}", vocab_size);
  fmt::println(fg(fmt::color::blue), "Input Size: {}", input_size);
  fmt::println(fg(fmt::color::blue), "Epoch: {}", epoch);
  fmt::println(fg(fmt::color::blue), "Learn Samples: {}", learn_samples);
  fmt::println(fg(fmt::color::blue), "Learning Repeat: {}", learning_set_repeats);
  fmt::println(fg(fmt::color::blue), "Learning Rate: {}", learning_rate);
  fmt::println(fg(fmt::color::blue), "Dynamic Learning Rate: {}", dynamic_lr);
  fmt::println(fg(fmt::color::blue), "Max Dynamic Learning Rate Reduction: {}", max_lr_reductions);
  fmt::println(fg(fmt::color::blue), "Balanced Learning: {}", balanced_learning);
  fmt::println(fg(fmt::color::blue), "Additional Info: {}", additional_acuracy_show);

  fmt::println(fg(fmt::color::yellow), "-- Prepering Data");
  std::string text_io = readLearningSet();
  std::string text = prepareLearningSet(text_io);
  std::vector<size_t> encoded_text = encodeTokens(text);
  if(encoded_text.size() <= context_size) return;
  if(tokens.size() != vocab_size) return;
  if(context_size == 0) return;

  fmt::println(fg(fmt::color::yellow), "-- Learning");
  fmt::println(fg(fmt::color::white), "enter to abort without lossing");

  setModel();

  std::array<float, input_size> input{};
  std::array<float, vocab_size> target{};
  std::array<size_t, context_size> active_input{};

  size_t previous_expected = 0;
  bool has_previous_sample = false;

  float best_ce = std::numeric_limits<float>::max();
  unsigned int stale_epochs = 0;
  unsigned int lr_reductions = 0;

  std::vector<std::vector<size_t>> token_positions(vocab_size);
  std::vector<size_t> available_tokens;
  std::vector<size_t> valid_positions;
  std::vector<size_t> learning_set(learn_samples);

  for(size_t i = 1; i < encoded_text.size(); i++){
    size_t token = encoded_text[i];
    if(token == unk_token || token == pad_token) continue;
    token_positions[token].push_back(i);
    valid_positions.push_back(i);
  };

  for(size_t i = 0; i < vocab_size; i++) if(!token_positions[i].empty()) available_tokens.push_back(i);
  if(valid_positions.empty()) return;

  for(unsigned int j = 0; j < epoch; j++){
    auto last = std::chrono::steady_clock::now();
    fmt::println("learn_samples: {}", learn_samples);
    fmt::println("learning_set_repeats: {}", learning_set_repeats);
    fmt::println("valid_positions: {}", valid_positions.size());

    float epoch_ce_sum = 0.0;
    size_t epoch_samples = 0;

    for(unsigned int i = 0; i < learn_samples; i++){
      if(balanced_learning){
        size_t token = available_tokens[rand() % available_tokens.size()];
        const std::vector<size_t>& positions = token_positions[token];
        learning_set[i] = positions[rand() % positions.size()];
      }
      else learning_set[i] = valid_positions[rand() % valid_positions.size()];
    };

    fmt::println("");
    fmt::println(fg(fmt::color::yellow), "-- Epoch {}/{}", j + 1, epoch);

    for(unsigned int r = 0; r < learning_set_repeats; r++){
      float ce_sum = 0.0;
      float mse_sum = 0.0;
      size_t correct = 0;
      size_t log_samples = 0;

      fmt::println(fg(fmt::color::white), "-- Repeat {}/{}", r + 1, learning_set_repeats);

      for(unsigned int i = 0; i < learn_samples; i++){
        if(hasInput()){
          std::string input_io;
          std::getline(std::cin, input_io);
          fmt::println("");
          fmt::println(fg(fmt::color::yellow), "-- Learning aborted");
          saveModelToFile();
          return;
        };

        if(has_previous_sample){
          for(size_t k = 0; k < context_size; k++) input[active_input[k]] = 0.0;
          target[previous_expected] = 0.0;
        };

        const size_t expected_position = learning_set[i];
        const size_t available = std::min(expected_position, context_size);
        const size_t words = 1 + rand() % available;
        const size_t padding = context_size - words;
        const size_t offset = expected_position - words;

        for(size_t k = 0; k < padding; k++){
          const size_t index = k * vocab_size + pad_token;
          active_input[k] = index;
          input[index] = 1.0;
        };

        for(size_t k = 0; k < words; k++){
          const size_t index = (padding + k) * vocab_size + encoded_text[offset + k];
          active_input[padding + k] = index;
          input[index] = 1.0;
        };

        const size_t expected = encoded_text[expected_position];
        target[expected] = 1.0;
        previous_expected = expected;
        has_previous_sample = true;

        model.setInput(input);
        model.forward();

        if(dynamic_lr){
          float* result = model.getActivatedResult();
          epoch_ce_sum += -std::log(std::max(result[expected], 0.0000001f));
          epoch_samples++;
        };

        if(additional_acuracy_show){
          const bool show_learning_ifno = ((i + 1) % 100 == 0 || i + 1 == learn_samples);
          learningInfo(show_learning_ifno, target, expected, ce_sum, mse_sum, correct, log_samples);
        };

        model.backprop(target);
        NN::Utils::progressBar(i, learn_samples);
      };
    };

    if(dynamic_lr && epoch_samples > 0){
      float epoch_ce = epoch_ce_sum / epoch_samples;

      if(epoch_ce < best_ce - 0.01){
        best_ce = epoch_ce;
        stale_epochs = 0;
      }
      else stale_epochs++;

      if(stale_epochs >= 3 && lr_reductions < max_lr_reductions){
        learning_rate *= 0.5f;
        learning_rate = std::max(learning_rate, 0.00001f);
        model.setLearningRate(learning_rate);
        best_ce = epoch_ce;
        stale_epochs = 0;
        lr_reductions++;
        fmt::println(fg(fmt::color::purple), "-- Learning rate reduced to {}", learning_rate);
      };
    };

    glFinish();

    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - last).count();
    
    fmt::println("Elapsed: {:.3f}s", elapsed);
    
    last = now;
    saveModelToFile();
  };

  saveModelToFile();
};



std::string LLM::LLM::getRespond(const std::string &message) noexcept {
  if(tokens.size() != vocab_size) return "";

  std::vector<size_t> context = encodeTokens(message);
  std::vector<size_t> response;
  response.reserve(response_size);

  setModel();

  std::array<float, input_size> input{};

  for(size_t i = 0; i < response_size; i++){
    input.fill(0.0);

    size_t words = std::min(context.size(), context_size);
    size_t padding = context_size - words;
    size_t start = context.size() - words;

    for(size_t k = 0; k < padding; k++) input[k * vocab_size + pad_token] = 1.0;

    for(size_t k = 0; k < words; k++) input[(padding + k) * vocab_size + context[start + k]] = 1.0;

    model.setInput(input);
    model.forward();

    float* result = model.getResult();
    size_t predicted = 0;

    for(size_t j = 1; j < vocab_size; j++) if(result[j] > result[predicted]) predicted = j;

    response.push_back(predicted);
    context.push_back(predicted);
  };

  return decodeTokens(response);
};



void LLM::LLM::printHelp() const noexcept {
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "Welcome to LLM");
  fmt::println(fg(fmt::color::blue), "help - see help message");
  fmt::println(fg(fmt::color::blue), "learn - to start learning from file");
  fmt::println(fg(fmt::color::blue), "set_learn_file - to set learn file");
  fmt::println(fg(fmt::color::blue), "set_learn_rate - to set learn rate");
  fmt::println(fg(fmt::color::blue), "set_learn_epoch - to set learn epoch");
  fmt::println(fg(fmt::color::blue), "set_learn_samples - to set learn samples");
  fmt::println(fg(fmt::color::blue), "set_learning_repeats - to set learn repeats");
  fmt::println(fg(fmt::color::blue), "set_balanced_learning - to set balanced learn");
  fmt::println(fg(fmt::color::blue), "set_learn_additional_log - to set learn logs");
  fmt::println(fg(fmt::color::blue), "set_dynamic_lr - to set dynamic learning rate");
  fmt::println(fg(fmt::color::blue), "print_model - print model");
  fmt::println(fg(fmt::color::blue), "quit - exit");
};



void LLM::LLM::onStart() noexcept {
  if(!loadTokensFromFile()){
    if(std::filesystem::exists(path_to_model_data)){
      fmt::println(fg(fmt::color::red), "Vocabulary missing or invalid for existing model");
      is_running = false;
      return;
    };

    std::string text = prepareLearningSet(readLearningSet());

    if(text.empty()){
      fmt::println(fg(fmt::color::red), "Learning set is empty");
      is_running = false;
      return;
    };

    createTokens(text);
    saveTokensToFile();
  };

  loadModelFromFile();
  printHelp();
};



void LLM::LLM::onUpdate() noexcept {
  std::string input = "";
  fmt::print(fg(fmt::color::white), "> ");
  std::getline(std::cin, input);

  if(input == "quit") { 
    is_running = false; 
    return; 
  };

  if(input == "help") {
    printHelp();
    return;
  };

  if(input == "learn") {
    learn();
    return;
  };

  if(input == "set_learn_file") {
    std::string file_name = "";
    fmt::print(fg(fmt::color::white), "file_name > ");
    std::getline(std::cin, file_name);
    
    if(!std::filesystem::exists(std::filesystem::path(__FILE__).parent_path() / std::filesystem::path(file_name)) || 
        std::filesystem::is_directory(std::filesystem::path(__FILE__).parent_path() / std::filesystem::path(file_name))){
      fmt::println(fg(fmt::color::red) | fmt::emphasis::bold, "{} didn't exist", file_name); 
      return;
    };
  
    setLearnFilePath(file_name);
    fmt::println(fg(fmt::color::green) | fmt::emphasis::bold, "Current Learn File: {}", getLearnFilePath().string()); 
    return;
  };

  if(input == "set_learn_rate"){
    std::string new_rate = "";
    fmt::print(fg(fmt::color::white), "new_rate > ");
    std::getline(std::cin, new_rate);
    
    float new_rate_val = 0;
    try{
      new_rate_val = std::stod(new_rate);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid float");
      return;
    };
    
    setLearningRate(new_rate_val);
    return;
  };

  if(input == "set_learn_epoch"){
    std::string new_epoch = "";
    fmt::print(fg(fmt::color::white), "new_epoch > ");
    std::getline(std::cin, new_epoch);
    
    int new_epoch_val = 0;
    try{
      new_epoch_val = std::stoi(new_epoch);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    
    setLearningEpoch(new_epoch_val);
    return;
  };

  if(input == "set_learn_samples"){
    std::string new_samples = "";
    fmt::print(fg(fmt::color::white), "new_samples > ");
    std::getline(std::cin, new_samples);
    
    int new_samples_val = 0;
    try{
      new_samples_val = std::stoi(new_samples);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    
    setLearningSamples(new_samples_val);
    return;
  };

  if(input == "set_learning_repeats"){
    std::string new_repeat = "";
    fmt::print(fg(fmt::color::white), "new_repeat > ");
    std::getline(std::cin, new_repeat);
    
    int new_repeat_val = 0;
    try{
      new_repeat_val = std::stoi(new_repeat);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    setLearningSetRepeats(new_repeat_val);
    return;
  };

  if(input == "set_learn_additional_log"){
    std::string new_samples = "";
    fmt::print(fg(fmt::color::white), "new_logs > ");
    std::getline(std::cin, new_samples);
    
    int new_samples_val = 0;
    try{
      new_samples_val = std::stoi(new_samples);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    
    setAdditionalAcuracyShow(new_samples_val);
    return;
  };

  if(input == "set_dynamic_lr"){
    std::string new_dynamic_lr = "";
    fmt::print(fg(fmt::color::white), "new_dynamic_lr > ");
    std::getline(std::cin, new_dynamic_lr);
    
    int new_new_dynamic_lr_val = 0;
    try{
      new_new_dynamic_lr_val = std::stoi(new_dynamic_lr);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    
    setDynamicLearningRate(new_new_dynamic_lr_val);
    return;
  };

  if(input == "set_balanced_learning"){
    std::string new_balanced = "";
    fmt::print(fg(fmt::color::white), "new_balanced > ");
    std::getline(std::cin, new_balanced);
    
    int new_new_balanced_val = 0;
    try{
      new_new_balanced_val = std::stoi(new_balanced);
    }
    catch(...){
      fmt::println(fg(fmt::color::red), "Invalid integer");
      return;
    };
    
    setBalancedLearning(new_new_balanced_val);
    return;
  };

  if(input == "print_model"){
    fmt::println(fg(fmt::color::blue), "{}", model.print());
    return;
  };

  std::string res = getRespond(input);
  fmt::println(fg(fmt::color::azure), "{}", res);
};



void LLM::LLM::onEnd() noexcept {
  saveModelToFile();
  saveTokensToFile();
};



void LLM::LLM::application() noexcept {
  onStart();
  
  if(!is_running) return;
  while(is_running) onUpdate();
  
  onEnd();
};
