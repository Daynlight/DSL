// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <memory>

#include "NeuralNetwork/Loss/Loss.h"



// =============================
// =========== MSE =============
// =============================
TEST(MSEFun, ReturnsZeroForEqualValues){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun(0.0f, 0.0f), 0.0f);
  EXPECT_FLOAT_EQ(mse.fun(1.0, 1.0f), 0.0f);
  EXPECT_FLOAT_EQ(mse.fun(-1.0, -1.0f), 0.0f);
  EXPECT_FLOAT_EQ(mse.fun(123.456, 123.456), 0.0f);
};

TEST(MSEFun, ReturnsExpectedLoss){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun(2.0, 1.0f), 0.5f);
  EXPECT_FLOAT_EQ(mse.fun(1.0, 2.0f), 0.5f);
  EXPECT_FLOAT_EQ(mse.fun(5.0, 1.0f), 8.0f);
  EXPECT_FLOAT_EQ(mse.fun(-2.0, 2.0f), 8.0f);
  EXPECT_FLOAT_EQ(mse.fun(-5.0, -1.0f), 8.0f);
};

TEST(MSEFun, IsSymmetric){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun(5.0, 2.0f), mse.fun(2.0, 5.0f));
  EXPECT_FLOAT_EQ(mse.fun(-5.0, 2.0f), mse.fun(2.0, -5.0f));
  EXPECT_FLOAT_EQ(mse.fun(-5.0, -2.0f), mse.fun(-2.0, -5.0f));
};

TEST(MSEFun, IsAlwaysNonNegative){
  NN::MSE mse;

  for(float x = -10.0f; x <= 10.0f; x += 0.5f)
    for(float t = -10.0f; t <= 10.0f; t += 0.5f)
      EXPECT_GE(mse.fun(x, t), 0.0f);
};

TEST(MSEFun, IncreasesWithDistance){
  NN::MSE mse;

  const float target = 0.0f;

  EXPECT_LT(mse.fun(1.0, target), mse.fun(2.0, target));
  EXPECT_LT(mse.fun(2.0, target), mse.fun(3.0, target));
  EXPECT_LT(mse.fun(-1.0, target), mse.fun(-2.0, target));
  EXPECT_LT(mse.fun(-2.0, target), mse.fun(-3.0, target));
};

TEST(MSEFun, HandlesLargeFiniteValues){
  NN::MSE mse;

  const float result = mse.fun(1e10f, 0.0f);

  EXPECT_TRUE(std::isfinite(result));
  EXPECT_FLOAT_EQ(result, 5e19f);
};

TEST(MSEFunPrime, ReturnsZeroForEqualValues){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun_prime(0.0f, 0.0f), 0.0f);
  EXPECT_FLOAT_EQ(mse.fun_prime(5.0, 5.0f), 0.0f);
  EXPECT_FLOAT_EQ(mse.fun_prime(-5.0, -5.0f), 0.0f);
};

TEST(MSEFunPrime, ReturnsExpectedDerivative){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun_prime(5.0, 2.0f), 3.0f);
  EXPECT_FLOAT_EQ(mse.fun_prime(2.0, 5.0f), -3.0f);
  EXPECT_FLOAT_EQ(mse.fun_prime(-2.0, 2.0f), -4.0f);
  EXPECT_FLOAT_EQ(mse.fun_prime(2.0, -2.0f), 4.0f);
};

TEST(MSEFunPrime, ChangesSignWhenArgumentsAreSwapped){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun_prime(5.0, 2.0f), -mse.fun_prime(2.0, 5.0f));
  EXPECT_FLOAT_EQ(mse.fun_prime(-5.0, 2.0f), -mse.fun_prime(2.0, -5.0f));
  EXPECT_FLOAT_EQ(mse.fun_prime(-5.0, -2.0f), -mse.fun_prime(-2.0, -5.0f));
};

TEST(MSEFunPrime, SignPointsAwayFromTarget){
  NN::MSE mse;

  EXPECT_GT(mse.fun_prime(5.0, 2.0f), 0.0f);
  EXPECT_LT(mse.fun_prime(2.0, 5.0f), 0.0f);
  EXPECT_GT(mse.fun_prime(1.0, 0.0f), 0.0f);
  EXPECT_LT(mse.fun_prime(-1.0, 0.0f), 0.0f);
};

TEST(MSEFunPrime, HandlesLargeFiniteValues){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun_prime(1e100, 0.0f), 1e100);
  EXPECT_FLOAT_EQ(mse.fun_prime(-1e100, 0.0f), -1e100);
};



// =============================
// ======== Interface ==========
// =============================
TEST(LossInterface, MSEWorksPolymorphically){
  std::unique_ptr<NN::iLoss> loss = std::make_unique<NN::MSE>();

  EXPECT_FLOAT_EQ(loss->fun(5.0, 2.0f), 4.5f);
  EXPECT_FLOAT_EQ(loss->fun_prime(5.0, 2.0f), 3.0f);
};