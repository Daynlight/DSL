// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <math.h>
#include <span>
#include <algorithm>



namespace NN{
class iActivation{
public:
  virtual float fun(std::span<const float> layer, size_t i) const noexcept = 0;
  virtual float fun_prime(std::span<const float> layer, size_t i, size_t j) const noexcept = 0;
};



class Linear : public iActivation{
public:
  float fun(std::span<const float> layer, size_t i) const noexcept;
  float fun_prime(std::span<const float> layer, size_t i, size_t j) const noexcept;
};

class Sigmoid : public iActivation{
public:
  float fun(std::span<const float> layer, size_t i) const noexcept;
  float fun_prime(std::span<const float> layer, size_t i, size_t j) const noexcept;
};

class Softmax : public iActivation{
public:
  float fun(std::span<const float> layer, size_t i) const noexcept;
  float fun_prime(std::span<const float> layer, size_t i, size_t j) const noexcept;
};

class ReLU : public iActivation{
public:
  float fun(std::span<const float> layer, size_t i) const noexcept;
  float fun_prime(std::span<const float> layer, size_t i, size_t j) const noexcept;
};
};



#include "Activation.hpp"
