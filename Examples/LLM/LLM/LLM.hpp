// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.




#include "LLM.h"




template<size_t V, size_t E>
inline LLM::Embedding<V, E>::Embedding() noexcept
  : data(V * E) {

  std::mt19937 generator(0xC0FFEEu);

  const float limit = std::sqrt(3.0f / static_cast<float>(E));
  std::uniform_real_distribution<float> distribution(-limit, limit);

  for(float& value : data) value = distribution(generator);
};



template<size_t V, size_t E>
inline LLM::Embedding<V, E>::~Embedding() noexcept {};



template<size_t V, size_t E>
inline void LLM::Embedding<V, E>::zero(size_t token) noexcept {
  if(token >= V) return;

  float* row = data.data() + token * E;

  for(size_t i = 0; i < E; i++) row[i] = 0.0f;
};



template<size_t V, size_t E>
template<size_t C>
inline void LLM::Embedding<V, E>::forward(const std::array<size_t, C>& tokens, std::array<float, C * E>& output) const noexcept {
  for(size_t i = 0; i < C; i++){
    const size_t token = tokens[i] < V ? tokens[i] : 0;
    const float* source = data.data() + token * E;
    float* destination = output.data() + i * E;

    std::memcpy(destination, source, E * sizeof(float));
  };
};



template<size_t V, size_t E>
template<size_t C>
inline void LLM::Embedding<V, E>::backprop(const std::array<size_t, C>& tokens, const float* gradient, float learning_rate, size_t frozen_token) noexcept {
  if(gradient == nullptr) return;

  for(size_t i = 0; i < C; i++){
    const size_t token = tokens[i];

    if(token >= V || token == frozen_token) continue;

    float* row = data.data() + token * E;
    const float* input_gradient = gradient + i * E;

    for(size_t j = 0; j < E; j++)
      row[j] -= learning_rate * input_gradient[j];
  };
};



template<size_t V, size_t E>
inline std::string LLM::Embedding<V, E>::serialize() const noexcept {
  std::string output(data.size() * sizeof(float), '\0');

  std::memcpy(output.data(), data.data(), output.size());

  return output;
};



template<size_t V, size_t E>
inline bool LLM::Embedding<V, E>::deserialize(std::string_view input) noexcept {
  if(input.size() != data.size() * sizeof(float)) return false;

  std::vector<float> new_data(data.size());

  std::memcpy(new_data.data(), input.data(), input.size());

  for(float value : new_data)
    if(!std::isfinite(value)) return false;

  data = std::move(new_data);

  return true;
};



template<size_t V, size_t E>
inline float* LLM::Embedding<V, E>::getData() noexcept {
  return data.data();
};



template<size_t V, size_t E>
inline const float* LLM::Embedding<V, E>::getData() const noexcept {
  return data.data();
};



template<size_t V, size_t E>
inline size_t LLM::Embedding<V, E>::size() const noexcept {
  return data.size();
};