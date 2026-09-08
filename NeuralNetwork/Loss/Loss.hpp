// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "Loss.h"



// ====================== //
// ========= MSE ======== //
// ====================== //
inline float NN::MSE::fun(float x, float t) const noexcept {
  return (x - t)*(x - t)/2;
};



inline float NN::MSE::fun_prime(float x, float t) const noexcept {
  return x - t;
};



// ====================== //
// ==== CrossEntropy ==== //
// ====================== //
inline float NN::CrossEntropy::fun(float x, float t) const noexcept {
  constexpr float epsilon = 0.0000001f;
  x = std::clamp(x, epsilon, 1.0f);
  return -t * std::log(x);
};



inline float NN::CrossEntropy::fun_prime(float x, float t) const noexcept {
  constexpr float epsilon = 0.0000001f;
  x = std::clamp(x, epsilon, 1.0f);
  return -t / x;
};