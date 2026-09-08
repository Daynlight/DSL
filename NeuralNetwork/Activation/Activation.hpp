// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "Activation.h"



// ===================== //
// ======= Linear ====== //
// ===================== //
inline float NN::Linear::fun(std::span<const float> layer, size_t i) const noexcept {
  return layer[i];
};



inline float NN::Linear::fun_prime(std::span<const float> layer, size_t i, size_t j) const noexcept {
  return i == j ? 1.0 : 0.0; 
};



// ====================== //
// ======= Sigmoid ====== //
// ====================== //
inline float NN::Sigmoid::fun(std::span<const float> layer, size_t i) const noexcept {
  if(layer[i] >= 0) return 1.0 / (1.0 + std::exp(-(layer[i])));
  else {
    float z = std::exp(layer[i]);
    return z / (1.0 + z);
  };
};



inline float NN::Sigmoid::fun_prime(std::span<const float> layer, size_t i, size_t j) const noexcept {
  if(i != j) return 0.0;
  float y = fun(layer, i);
  return y * (1 - y); 
};



// ====================== //
// ======= Softmax ====== //
// ====================== //
inline float NN::Softmax::fun(std::span<const float> layer, size_t i) const noexcept {
  float max = *std::max_element(layer.begin(), layer.end());
  float sum = 0.0;

  for(float x : layer) sum += std::exp(x - max);

  return std::exp(layer[i] - max) / sum;
};



inline float NN::Softmax::fun_prime(std::span<const float> layer, size_t i, size_t j) const noexcept {
  float pi = fun(layer, i);
  float pj = fun(layer, j);

  if(i == j) return pi * (1.0 - pi);

  return -pi * pj;
};



// ====================== //
// ======== ReLU ======== //
// ====================== //
inline float NN::ReLU::fun(std::span<const float> layer, size_t i) const noexcept {
  return std::max(0.0f, layer[i]);
};



inline float NN::ReLU::fun_prime(std::span<const float> layer, size_t i, size_t j) const noexcept {
  if(i != j) return 0.0;
  return layer[i] > 0.0 ? 1.0 : 0.0;
};
