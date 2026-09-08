// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <algorithm>
#include <cmath>



namespace NN{
class iLoss{
public:
  virtual float fun(float x, float t) const noexcept = 0;
  virtual float fun_prime(float x, float t) const noexcept = 0;
};



class MSE : public iLoss{
public:
  float fun(float x, float t) const noexcept;
  float fun_prime(float x, float t) const noexcept;
};

class CrossEntropy : public iLoss{
public:
  float fun(float x, float t) const noexcept;
  float fun_prime(float x, float t) const noexcept;
};
};



#include "Loss.hpp"
