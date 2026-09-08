// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>

#include "NeuralNetwork/Loss/Loss.h"



// =============================
// ======== Interface ==========
// =============================
TEST(LossInterface, MSEWorksPolymorphically){
  std::unique_ptr<NN::iLoss> loss = std::make_unique<NN::MSE>();

  EXPECT_FLOAT_EQ(loss->fun(5.0f, 2.0f), 4.5f);
  EXPECT_FLOAT_EQ(loss->fun_prime(5.0f, 2.0f), 3.0f);
};

TEST(LossInterface, CrossEntropyWorksPolymorphically){
  std::unique_ptr<NN::iLoss> loss = std::make_unique<NN::CrossEntropy>();

  EXPECT_NEAR(loss->fun(0.5f, 1.0f), std::log(2.0), 1e-6);
  EXPECT_FLOAT_EQ(loss->fun_prime(0.5f, 1.0f), -2.0f);
};



// =============================
// =========== MSE =============
// =============================
TEST(MSEFun, ReturnsZeroForEqualValues){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun(0.0f, 0.0f), 0.0f);
  EXPECT_FLOAT_EQ(mse.fun(1.0f, 1.0f), 0.0f);
  EXPECT_FLOAT_EQ(mse.fun(-1.0f, -1.0f), 0.0f);
  EXPECT_FLOAT_EQ(mse.fun(123.456f, 123.456f), 0.0f);
};

TEST(MSEFun, ReturnsExpectedLoss){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun(2.0f, 1.0f), 0.5f);
  EXPECT_FLOAT_EQ(mse.fun(1.0f, 2.0f), 0.5f);
  EXPECT_FLOAT_EQ(mse.fun(5.0f, 1.0f), 8.0f);
  EXPECT_FLOAT_EQ(mse.fun(-2.0f, 2.0f), 8.0f);
  EXPECT_FLOAT_EQ(mse.fun(-5.0f, -1.0f), 8.0f);
};

TEST(MSEFun, IsSymmetric){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun(5.0f, 2.0f), mse.fun(2.0f, 5.0f));
  EXPECT_FLOAT_EQ(mse.fun(-5.0f, 2.0f), mse.fun(2.0f, -5.0f));
  EXPECT_FLOAT_EQ(mse.fun(-5.0f, -2.0f), mse.fun(-2.0f, -5.0f));
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

  EXPECT_LT(mse.fun(1.0f, target), mse.fun(2.0f, target));
  EXPECT_LT(mse.fun(2.0f, target), mse.fun(3.0f, target));
  EXPECT_LT(mse.fun(-1.0f, target), mse.fun(-2.0f, target));
  EXPECT_LT(mse.fun(-2.0f, target), mse.fun(-3.0f, target));
};

TEST(MSEFun, HandlesLargeFiniteValues){
  NN::MSE mse;

  const float result = mse.fun(1e10f, 0.0f);

  EXPECT_TRUE(std::isfinite(result));
  EXPECT_FLOAT_EQ(result, 5e19f);
};

TEST(MSEFun, PropagatesNaN){
  NN::MSE mse;

  const float nan = std::numeric_limits<float>::quiet_NaN();

  EXPECT_TRUE(std::isnan(mse.fun(nan, 0.0f)));
  EXPECT_TRUE(std::isnan(mse.fun(0.0f, nan)));
};

TEST(MSEFunPrime, ReturnsZeroForEqualValues){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun_prime(0.0f, 0.0f), 0.0f);
  EXPECT_FLOAT_EQ(mse.fun_prime(5.0f, 5.0f), 0.0f);
  EXPECT_FLOAT_EQ(mse.fun_prime(-5.0f, -5.0f), 0.0f);
};

TEST(MSEFunPrime, ReturnsExpectedDerivative){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun_prime(5.0f, 2.0f), 3.0f);
  EXPECT_FLOAT_EQ(mse.fun_prime(2.0f, 5.0f), -3.0f);
  EXPECT_FLOAT_EQ(mse.fun_prime(-2.0f, 2.0f), -4.0f);
  EXPECT_FLOAT_EQ(mse.fun_prime(2.0f, -2.0f), 4.0f);
};

TEST(MSEFunPrime, ChangesSignWhenArgumentsAreSwapped){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun_prime(5.0f, 2.0f), -mse.fun_prime(2.0f, 5.0f));
  EXPECT_FLOAT_EQ(mse.fun_prime(-5.0f, 2.0f), -mse.fun_prime(2.0f, -5.0f));
  EXPECT_FLOAT_EQ(mse.fun_prime(-5.0f, -2.0f), -mse.fun_prime(-2.0f, -5.0f));
};

TEST(MSEFunPrime, SignPointsAwayFromTarget){
  NN::MSE mse;

  EXPECT_GT(mse.fun_prime(5.0f, 2.0f), 0.0f);
  EXPECT_LT(mse.fun_prime(2.0f, 5.0f), 0.0f);
  EXPECT_GT(mse.fun_prime(1.0f, 0.0f), 0.0f);
  EXPECT_LT(mse.fun_prime(-1.0f, 0.0f), 0.0f);
};

TEST(MSEFunPrime, HandlesLargeFiniteValues){
  NN::MSE mse;

  EXPECT_FLOAT_EQ(mse.fun_prime(1e10f, 0.0f), 1e10f);
  EXPECT_FLOAT_EQ(mse.fun_prime(-1e10f, 0.0f), -1e10f);
};

TEST(MSEFunPrime, PropagatesNaN){
  NN::MSE mse;

  const float nan = std::numeric_limits<float>::quiet_NaN();

  EXPECT_TRUE(std::isnan(mse.fun_prime(nan, 0.0f)));
  EXPECT_TRUE(std::isnan(mse.fun_prime(0.0f, nan)));
};



// =============================
// ======== CrossEntropy =======
// =============================
TEST(CrossEntropyFun, ReturnsZeroForCorrectPrediction){
  NN::CrossEntropy ce;

  EXPECT_FLOAT_EQ(ce.fun(1.0f, 1.0f), 0.0f);
  EXPECT_FLOAT_EQ(ce.fun(0.5f, 0.0f), 0.0f);
  EXPECT_FLOAT_EQ(ce.fun(0.1f, 0.0f), 0.0f);
};

TEST(CrossEntropyFun, ReturnsExpectedLoss){
  NN::CrossEntropy ce;

  EXPECT_NEAR(ce.fun(0.5f, 1.0f), std::log(2.0), 1e-6);
  EXPECT_NEAR(ce.fun(0.25f, 1.0f), std::log(4.0), 1e-6);
  EXPECT_NEAR(ce.fun(0.5f, 0.5f), std::log(2.0) / 2.0, 1e-6);
  EXPECT_NEAR(ce.fun(0.25f, 0.5f), std::log(4.0) / 2.0, 1e-6);
};

TEST(CrossEntropyFun, IsAlwaysNonNegativeForValidInputs){
  NN::CrossEntropy ce;

  for(float x = 0.0f; x <= 1.0f; x += 0.01f)
    for(float t = 0.0f; t <= 1.0f; t += 0.1f)
      EXPECT_GE(ce.fun(x, t), 0.0f);
};

TEST(CrossEntropyFun, DecreasesAsCorrectPredictionIncreases){
  NN::CrossEntropy ce;

  EXPECT_GT(ce.fun(0.1f, 1.0f), ce.fun(0.2f, 1.0f));
  EXPECT_GT(ce.fun(0.2f, 1.0f), ce.fun(0.5f, 1.0f));
  EXPECT_GT(ce.fun(0.5f, 1.0f), ce.fun(0.9f, 1.0f));
  EXPECT_GT(ce.fun(0.9f, 1.0f), ce.fun(1.0f, 1.0f));
};

TEST(CrossEntropyFun, ClampsSmallProbabilities){
  NN::CrossEntropy ce;

  const float expected = -std::log(1e-7f);

  EXPECT_NEAR(ce.fun(0.0f, 1.0f), expected, 1e-6);
  EXPECT_NEAR(ce.fun(-1.0f, 1.0f), expected, 1e-6);
  EXPECT_NEAR(ce.fun(1e-10f, 1.0f), expected, 1e-6);
  EXPECT_NEAR(ce.fun(1e-7f, 1.0f), expected, 1e-6);
};

TEST(CrossEntropyFun, ClampsProbabilitiesAboveOne){
  NN::CrossEntropy ce;

  EXPECT_FLOAT_EQ(ce.fun(1.0f, 1.0f), 0.0f);
  EXPECT_FLOAT_EQ(ce.fun(2.0f, 1.0f), 0.0f);
  EXPECT_FLOAT_EQ(ce.fun(1000.0f, 1.0f), 0.0f);
};

TEST(CrossEntropyFun, ZeroTargetProducesZeroForFiniteInputs){
  NN::CrossEntropy ce;

  EXPECT_FLOAT_EQ(ce.fun(0.0f, 0.0f), 0.0f);
  EXPECT_FLOAT_EQ(ce.fun(0.25f, 0.0f), 0.0f);
  EXPECT_FLOAT_EQ(ce.fun(1.0f, 0.0f), 0.0f);
  EXPECT_FLOAT_EQ(ce.fun(2.0f, 0.0f), 0.0f);
};

TEST(CrossEntropyFun, IsLinearInTarget){
  NN::CrossEntropy ce;

  EXPECT_NEAR(ce.fun(0.25f, 0.5f), 0.5f * ce.fun(0.25f, 1.0f), 1e-6);
  EXPECT_NEAR(ce.fun(0.5f, 0.25f), 0.25f * ce.fun(0.5f, 1.0f), 1e-6);
};

TEST(CrossEntropyFunPrime, ReturnsExpectedDerivative){
  NN::CrossEntropy ce;

  EXPECT_FLOAT_EQ(ce.fun_prime(0.5f, 1.0f), -2.0f);
  EXPECT_FLOAT_EQ(ce.fun_prime(0.25f, 1.0f), -4.0f);
  EXPECT_FLOAT_EQ(ce.fun_prime(0.5f, 0.5f), -1.0f);
  EXPECT_FLOAT_EQ(ce.fun_prime(1.0f, 1.0f), -1.0f);
};

TEST(CrossEntropyFunPrime, ReturnsZeroForZeroTarget){
  NN::CrossEntropy ce;

  EXPECT_FLOAT_EQ(ce.fun_prime(0.0f, 0.0f), 0.0f);
  EXPECT_FLOAT_EQ(ce.fun_prime(0.5f, 0.0f), 0.0f);
  EXPECT_FLOAT_EQ(ce.fun_prime(1.0f, 0.0f), 0.0f);
};

TEST(CrossEntropyFunPrime, IsNegativeForPositiveTarget){
  NN::CrossEntropy ce;

  for(float x = 0.1f; x <= 1.0f; x += 0.1f)
    EXPECT_LT(ce.fun_prime(x, 1.0f), 0.0f);
};

TEST(CrossEntropyFunPrime, ClampsSmallProbabilities){
  NN::CrossEntropy ce;

  EXPECT_FLOAT_EQ(ce.fun_prime(0.0f, 1.0f), -1e7f);
  EXPECT_FLOAT_EQ(ce.fun_prime(-1.0f, 1.0f), -1e7f);
  EXPECT_FLOAT_EQ(ce.fun_prime(1e-10f, 1.0f), -1e7f);
  EXPECT_FLOAT_EQ(ce.fun_prime(1e-7f, 1.0f), -1e7f);
};

TEST(CrossEntropyFunPrime, ClampsProbabilitiesAboveOne){
  NN::CrossEntropy ce;

  EXPECT_FLOAT_EQ(ce.fun_prime(1.0f, 1.0f), -1.0f);
  EXPECT_FLOAT_EQ(ce.fun_prime(2.0f, 1.0f), -1.0f);
  EXPECT_FLOAT_EQ(ce.fun_prime(1000.0f, 1.0f), -1.0f);
};

TEST(CrossEntropyFunPrime, IsLinearInTarget){
  NN::CrossEntropy ce;

  EXPECT_NEAR(ce.fun_prime(0.25f, 0.5f), 0.5f * ce.fun_prime(0.25f, 1.0f), 1e-6);
  EXPECT_NEAR(ce.fun_prime(0.5f, 0.25f), 0.25f * ce.fun_prime(0.5f, 1.0f), 1e-6);
};

TEST(CrossEntropyFunPrime, MagnitudeIncreasesAsProbabilityDecreases){
  NN::CrossEntropy ce;

  EXPECT_GT(std::abs(ce.fun_prime(0.1f, 1.0f)), std::abs(ce.fun_prime(0.2f, 1.0f)));
  EXPECT_GT(std::abs(ce.fun_prime(0.2f, 1.0f)), std::abs(ce.fun_prime(0.5f, 1.0f)));
  EXPECT_GT(std::abs(ce.fun_prime(0.5f, 1.0f)), std::abs(ce.fun_prime(1.0f, 1.0f)));
};



// =============================
// ======= Mathematics =========
// =============================
template<typename Loss>
void checkLossDerivative(const std::array<std::array<float, 2>, 6>& samples){
  Loss loss;

  constexpr float epsilon = 0.001f;
  constexpr double tolerance = 0.0005;

  for(size_t i = 0; i < samples.size(); i++){
    float x = samples[i][0];
    float t = samples[i][1];

    double positive = loss.fun(x + epsilon, t);
    double negative = loss.fun(x - epsilon, t);
    double numerical = (positive - negative) / double((x + epsilon) - (x - epsilon));
    double analytical = loss.fun_prime(x, t);

    SCOPED_TRACE(::testing::Message() << "x=" << x << " t=" << t);

    ASSERT_TRUE(std::isfinite(numerical));
    ASSERT_TRUE(std::isfinite(analytical));
    EXPECT_NEAR(analytical, numerical, tolerance);
  };
};

TEST(MSEMathematics, NumericalDerivative){
  std::array<std::array<float, 2>, 6> samples = {{
    {-3.0f, 1.0f},
    {-1.0f, -2.0f},
    {0.0f, 0.0f},
    {0.5f, 1.0f},
    {2.0f, -1.0f},
    {5.0f, 2.0f}
  }};

  checkLossDerivative<NN::MSE>(samples);
};

TEST(CrossEntropyMathematics, NumericalDerivative){
  std::array<std::array<float, 2>, 6> samples = {{
    {0.1f, 1.0f},
    {0.25f, 0.5f},
    {0.4f, 1.0f},
    {0.5f, 0.0f},
    {0.7f, 0.25f},
    {0.9f, 1.0f}
  }};

  checkLossDerivative<NN::CrossEntropy>(samples);
};

TEST(CrossEntropyMathematics, ClippedGradientContract){
  NN::CrossEntropy ce;

  constexpr float epsilon = 1e-7f;

  EXPECT_FLOAT_EQ(ce.fun_prime(0.0f, 1.0f), -1.0f / epsilon);
  EXPECT_FLOAT_EQ(ce.fun_prime(-1.0f, 1.0f), -1.0f / epsilon);
  EXPECT_FLOAT_EQ(ce.fun_prime(2.0f, 1.0f), -1.0f);
};



// =============================================================================
// ================================= Shaders ===================================
// =============================================================================



#include "NeuralNetwork/GPUAcceleration/GPUAcceleration.h"
#include "CWindow/Renderer/OpenGL/Shader/SSBO/Shader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>



struct LossShaderBuffers {
  std::vector<CW::Renderer::GPUStore> storage;
  size_t size;

  LossShaderBuffers(const std::vector<float>& activated, const std::vector<float>& target)
    : storage(7), size(activated.size()) {
    NN::GPUAcceleration::get();
    storage[3].set(activated);
    storage[5].set(std::vector<float>(size, 0.0f));
    storage[6].set(target);
  };

  void run(CW::Renderer::ComputeShader& shader, unsigned int groups){
    storage[5].set(std::vector<float>(size, std::numeric_limits<float>::quiet_NaN()));
    shader.run(storage, groups);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  };

  std::vector<float> get(size_t binding){
    std::vector<float> result;
    storage[binding].get(result);
    return result;
  };
};



// =============================
// ======== MSE Shader =========
// =============================
TEST(MSEShader, ReturnsExpectedValues){
  std::vector<float> input = {0.0f, 2.0f, 1.0f, 5.0f, -2.0f};
  std::vector<float> target = {0.0f, 1.0f, 2.0f, 1.0f, 2.0f};

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossMSEShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], 0.5f);
  EXPECT_FLOAT_EQ(result[2], 0.5f);
  EXPECT_FLOAT_EQ(result[3], 8.0f);
  EXPECT_FLOAT_EQ(result[4], 8.0f);
};

TEST(MSEShader, HandlesLargeFiniteValues){
  std::vector<float> input = {1e10f, -1e10f};
  std::vector<float> target = {0.0f, 0.0f};

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossMSEShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_TRUE(std::isfinite(result[0]));
  EXPECT_TRUE(std::isfinite(result[1]));
  EXPECT_FLOAT_EQ(result[0], 5e19f);
  EXPECT_FLOAT_EQ(result[1], 5e19f);
};

TEST(MSEShader, HandlesWorkgroupBoundaries){
  auto& shader = NN::GPUAcceleration::get().getLossMSEShader();

  for(size_t size : {1u, 255u, 256u, 257u, 513u}){
    std::vector<float> input(size);
    std::vector<float> target(size);

    for(size_t i = 0; i < size; i++){
      input[i] = float(int(i % 21) - 10) * 0.25f;
      target[i] = float(int(i % 7) - 3) * 0.5f;
    };

    LossShaderBuffers buffers(input, target);

    buffers.run(shader, (size + 255) / 256);

    std::vector<float> result = buffers.get(5);

    ASSERT_EQ(result.size(), size);

    for(size_t i = 0; i < size; i++){
      double difference = double(input[i]) - target[i];
      double expected = difference * difference / 2.0;

      SCOPED_TRACE(::testing::Message() << "size=" << size << " i=" << i);
      ASSERT_TRUE(std::isfinite(result[i]));
      EXPECT_NEAR(result[i], expected, 1e-6);
    };
  };
};



// =============================
// ===== MSE Prime Shader ======
// =============================
TEST(MSEPrimeShader, ReturnsExpectedValues){
  std::vector<float> input = {0.0f, 5.0f, 2.0f, -2.0f, 2.0f};
  std::vector<float> target = {0.0f, 2.0f, 5.0f, 2.0f, -2.0f};

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossMSEPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], 3.0f);
  EXPECT_FLOAT_EQ(result[2], -3.0f);
  EXPECT_FLOAT_EQ(result[3], -4.0f);
  EXPECT_FLOAT_EQ(result[4], 4.0f);
};

TEST(MSEPrimeShader, HandlesLargeFiniteValues){
  std::vector<float> input = {1e10f, -1e10f};
  std::vector<float> target = {0.0f, 0.0f};

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossMSEPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 1e10f);
  EXPECT_FLOAT_EQ(result[1], -1e10f);
};

TEST(MSEPrimeShader, HandlesWorkgroupBoundaries){
  auto& shader = NN::GPUAcceleration::get().getLossMSEPrimeShader();

  for(size_t size : {1u, 255u, 256u, 257u, 513u}){
    std::vector<float> input(size);
    std::vector<float> target(size);

    for(size_t i = 0; i < size; i++){
      input[i] = float(int(i % 21) - 10) * 0.25f;
      target[i] = float(int(i % 7) - 3) * 0.5f;
    };

    LossShaderBuffers buffers(input, target);

    buffers.run(shader, (size + 255) / 256);

    std::vector<float> result = buffers.get(5);

    ASSERT_EQ(result.size(), size);

    for(size_t i = 0; i < size; i++){
      double expected = double(input[i]) - target[i];

      SCOPED_TRACE(::testing::Message() << "size=" << size << " i=" << i);
      ASSERT_TRUE(std::isfinite(result[i]));
      EXPECT_NEAR(result[i], expected, 1e-6);
    };
  };
};



// =============================
// ==== CrossEntropy Shader ====
// =============================
TEST(CrossEntropyShader, ReturnsExpectedValues){
  std::vector<float> input = {1.0f, 0.5f, 0.25f, 0.5f};
  std::vector<float> target = {1.0f, 1.0f, 1.0f, 0.5f};

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossCrossEntropyShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_NEAR(result[1], std::log(2.0), 1e-6);
  EXPECT_NEAR(result[2], std::log(4.0), 1e-6);
  EXPECT_NEAR(result[3], std::log(2.0) / 2.0, 1e-6);
};

TEST(CrossEntropyShader, ClampsSmallProbabilities){
  std::vector<float> input = {0.0f, -1.0f, 1e-10f, 1e-7f};
  std::vector<float> target(input.size(), 1.0f);

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossCrossEntropyShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(float value : result){
    ASSERT_TRUE(std::isfinite(value));
    EXPECT_NEAR(value, -std::log(1e-7f), 1e-6);
  };
};

TEST(CrossEntropyShader, ClampsProbabilitiesAboveOne){
  std::vector<float> input = {1.0f, 2.0f, 1000.0f};
  std::vector<float> target(input.size(), 1.0f);

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossCrossEntropyShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(float value : result)
    EXPECT_FLOAT_EQ(value, 0.0f);
};

TEST(CrossEntropyShader, ZeroTargetProducesZero){
  std::vector<float> input = {0.0f, 0.25f, 1.0f, 2.0f};
  std::vector<float> target(input.size(), 0.0f);

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossCrossEntropyShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(float value : result)
    EXPECT_FLOAT_EQ(value, 0.0f);
};

TEST(CrossEntropyShader, HandlesWorkgroupBoundaries){
  auto& shader = NN::GPUAcceleration::get().getLossCrossEntropyShader();

  for(size_t size : {1u, 255u, 256u, 257u, 513u}){
    std::vector<float> input(size);
    std::vector<float> target(size);

    for(size_t i = 0; i < size; i++){
      input[i] = float(i % 9 + 1) / 10.0f;
      target[i] = float(i % 5) / 4.0f;
    };

    LossShaderBuffers buffers(input, target);

    buffers.run(shader, (size + 255) / 256);

    std::vector<float> result = buffers.get(5);

    ASSERT_EQ(result.size(), size);

    for(size_t i = 0; i < size; i++){
      double expected = -double(target[i]) * std::log(double(input[i]));

      SCOPED_TRACE(::testing::Message() << "size=" << size << " i=" << i);
      ASSERT_TRUE(std::isfinite(result[i]));
      EXPECT_NEAR(result[i], expected, 1e-5);
    };
  };
};



// =============================
// = CrossEntropy Prime Shader =
// =============================
TEST(CrossEntropyPrimeShader, ReturnsExpectedValues){
  std::vector<float> input = {0.5f, 0.25f, 0.5f, 1.0f};
  std::vector<float> target = {1.0f, 1.0f, 0.5f, 1.0f};

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossCrossEntropyPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], -2.0f);
  EXPECT_FLOAT_EQ(result[1], -4.0f);
  EXPECT_FLOAT_EQ(result[2], -1.0f);
  EXPECT_FLOAT_EQ(result[3], -1.0f);
};

TEST(CrossEntropyPrimeShader, ReturnsZeroForZeroTarget){
  std::vector<float> input = {0.0f, 0.25f, 1.0f, 2.0f};
  std::vector<float> target(input.size(), 0.0f);

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossCrossEntropyPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(float value : result)
    EXPECT_FLOAT_EQ(value, 0.0f);
};

TEST(CrossEntropyPrimeShader, ClampsSmallProbabilities){
  std::vector<float> input = {0.0f, -1.0f, 1e-10f, 1e-7f};
  std::vector<float> target(input.size(), 1.0f);

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossCrossEntropyPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(float value : result){
    ASSERT_TRUE(std::isfinite(value));
    EXPECT_NEAR(value, -1e7f, 1.0f);
  };
};

TEST(CrossEntropyPrimeShader, ClampsProbabilitiesAboveOne){
  std::vector<float> input = {1.0f, 2.0f, 1000.0f};
  std::vector<float> target(input.size(), 1.0f);

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossCrossEntropyPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(float value : result)
    EXPECT_FLOAT_EQ(value, -1.0f);
};

TEST(CrossEntropyPrimeShader, HandlesWorkgroupBoundaries){
  auto& shader = NN::GPUAcceleration::get().getLossCrossEntropyPrimeShader();

  for(size_t size : {1u, 255u, 256u, 257u, 513u}){
    std::vector<float> input(size);
    std::vector<float> target(size);

    for(size_t i = 0; i < size; i++){
      input[i] = float(i % 9 + 1) / 10.0f;
      target[i] = float(i % 5) / 4.0f;
    };

    LossShaderBuffers buffers(input, target);

    buffers.run(shader, (size + 255) / 256);

    std::vector<float> result = buffers.get(5);

    ASSERT_EQ(result.size(), size);

    for(size_t i = 0; i < size; i++){
      double expected = -double(target[i]) / input[i];

      SCOPED_TRACE(::testing::Message() << "size=" << size << " i=" << i);
      ASSERT_TRUE(std::isfinite(result[i]));
      EXPECT_NEAR(result[i], expected, 1e-5);
    };
  };
};



// =============================
// ===== Shader Mathematics ====
// =============================
void checkLossShaderDerivative(CW::Renderer::ComputeShader& shader, CW::Renderer::ComputeShader& prime, const std::vector<float>& input, const std::vector<float>& target){
  LossShaderBuffers buffers(input, target);

  const unsigned int groups = (input.size() + 255) / 256;

  buffers.run(prime, groups);

  std::vector<float> analytical = buffers.get(5);

  ASSERT_EQ(analytical.size(), input.size());

  constexpr float epsilon = 0.001f;
  constexpr double tolerance = 0.001;

  for(size_t j = 0; j < input.size(); j++){
    auto plus = input;
    auto minus = input;

    plus[j] += epsilon;
    minus[j] -= epsilon;

    buffers.storage[3].set(plus);
    buffers.run(shader, groups);
    std::vector<float> positive = buffers.get(5);

    buffers.storage[3].set(minus);
    buffers.run(shader, groups);
    std::vector<float> negative = buffers.get(5);

    ASSERT_EQ(positive.size(), input.size());
    ASSERT_EQ(negative.size(), input.size());

    for(size_t i = 0; i < input.size(); i++){
      double numerical = (double(positive[i]) - negative[i]) / double(plus[j] - minus[j]);
      double expected = i == j ? analytical[i] : 0.0;

      SCOPED_TRACE(::testing::Message() << "i=" << i << " j=" << j);

      ASSERT_TRUE(std::isfinite(numerical));
      ASSERT_TRUE(std::isfinite(expected));
      EXPECT_NEAR(expected, numerical, tolerance);
    };
  };
};

TEST(MSEShaderMathematics, NumericalDerivative){
  std::vector<float> input = {-3.0f, -1.0f, 0.0f, 0.5f, 2.0f, 5.0f};
  std::vector<float> target = {1.0f, -2.0f, 0.0f, 1.0f, -1.0f, 2.0f};

  auto& gpu = NN::GPUAcceleration::get();

  checkLossShaderDerivative(gpu.getLossMSEShader(), gpu.getLossMSEPrimeShader(), input, target);
};

TEST(CrossEntropyShaderMathematics, NumericalDerivative){
  std::vector<float> input = {0.1f, 0.25f, 0.4f, 0.5f, 0.7f, 0.9f};
  std::vector<float> target = {1.0f, 0.5f, 1.0f, 0.0f, 0.25f, 1.0f};

  auto& gpu = NN::GPUAcceleration::get();

  checkLossShaderDerivative(gpu.getLossCrossEntropyShader(), gpu.getLossCrossEntropyPrimeShader(), input, target);
};

TEST(CrossEntropyShaderMathematics, ClippedGradientContract){
  std::vector<float> input = {0.0f, -1.0f, 1e-10f, 1.0f, 2.0f};
  std::vector<float> target(input.size(), 1.0f);

  LossShaderBuffers buffers(input, target);

  buffers.run(NN::GPUAcceleration::get().getLossCrossEntropyPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_NEAR(result[0], -1e7f, 1.0f);
  EXPECT_NEAR(result[1], -1e7f, 1.0f);
  EXPECT_NEAR(result[2], -1e7f, 1.0f);
  EXPECT_FLOAT_EQ(result[3], -1.0f);
  EXPECT_FLOAT_EQ(result[4], -1.0f);
};