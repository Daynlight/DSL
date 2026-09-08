// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>

#include "NeuralNetwork/Activation/Activation.h"



// =============================
// ======= Interface ===========
// =============================
TEST(ActivationInterface, LinearWorksPolymorphically){
  std::unique_ptr<NN::iActivation> activation = std::make_unique<NN::Linear>();

  std::array<float, 2> layer = {2.5, 5.0};

  EXPECT_FLOAT_EQ(activation->fun(layer, 0), 2.5);
  EXPECT_FLOAT_EQ(activation->fun_prime(layer, 0, 0), 1.0);
  EXPECT_FLOAT_EQ(activation->fun_prime(layer, 0, 1), 0.0);
};

TEST(ActivationInterface, SigmoidWorksPolymorphically){
  std::unique_ptr<NN::iActivation> activation = std::make_unique<NN::Sigmoid>();

  std::array<float, 2> layer = {1.0, -1.0};

  EXPECT_NEAR(activation->fun(layer, 0), 0.7310585786300049, 1e-6);
  EXPECT_NEAR(activation->fun_prime(layer, 0, 0), 0.19661193324148185, 1e-6);
  EXPECT_FLOAT_EQ(activation->fun_prime(layer, 0, 1), 0.0);
};

TEST(ActivationInterface, SoftmaxWorksPolymorphically){
  std::unique_ptr<NN::iActivation> activation = std::make_unique<NN::Softmax>();

  std::array<float, 3> layer = {1.0, 2.0, 3.0};

  float sum = 0.0;

  for(size_t i = 0; i < layer.size(); i++)
    sum += activation->fun(layer, i);

  EXPECT_NEAR(sum, 1.0, 1e-6);
};



// =============================
// ========= Linear ============
// =============================
TEST(LinearFun, ReturnsInput){
  NN::Linear linear;

  std::array<float, 5> layer = {0.0, 1.0, -1.0, 123.456, -123.456};

  EXPECT_FLOAT_EQ(linear.fun(layer, 0), 0.0);
  EXPECT_FLOAT_EQ(linear.fun(layer, 1), 1.0);
  EXPECT_FLOAT_EQ(linear.fun(layer, 2), -1.0);
  EXPECT_FLOAT_EQ(linear.fun(layer, 3), 123.456);
  EXPECT_FLOAT_EQ(linear.fun(layer, 4), -123.456);
};

TEST(LinearFun, HandlesExtremeValues){
  NN::Linear linear;

  std::array<float, 2> layer = {
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::lowest()
  };

  EXPECT_FLOAT_EQ(linear.fun(layer, 0), std::numeric_limits<float>::max());
  EXPECT_FLOAT_EQ(linear.fun(layer, 1), std::numeric_limits<float>::lowest());
};

TEST(LinearFunPrime, ReturnsIdentityJacobian){
  NN::Linear linear;

  std::array<float, 3> layer = {1.0, 2.0, 3.0};

  EXPECT_FLOAT_EQ(linear.fun_prime(layer, 0, 0), 1.0);
  EXPECT_FLOAT_EQ(linear.fun_prime(layer, 1, 1), 1.0);
  EXPECT_FLOAT_EQ(linear.fun_prime(layer, 2, 2), 1.0);

  EXPECT_FLOAT_EQ(linear.fun_prime(layer, 0, 1), 0.0);
  EXPECT_FLOAT_EQ(linear.fun_prime(layer, 0, 2), 0.0);
  EXPECT_FLOAT_EQ(linear.fun_prime(layer, 1, 0), 0.0);
  EXPECT_FLOAT_EQ(linear.fun_prime(layer, 2, 1), 0.0);
};

TEST(LinearFun, PropagatesNaN){
  NN::Linear linear;

  std::array<float, 1> layer = {std::numeric_limits<float>::quiet_NaN()};

  EXPECT_TRUE(std::isnan(linear.fun(layer, 0)));
};

TEST(LinearFunPrime, IsIndependentOfInput){
  NN::Linear linear;

  std::array<float, 3> layer = {-1000.0f, 0.0f, 1000.0f};

  for(size_t i = 0; i < layer.size(); i++){
    for(size_t j = 0; j < layer.size(); j++)
      EXPECT_FLOAT_EQ(linear.fun_prime(layer, i, j), i == j ? 1.0f : 0.0f);
  };
};



// =============================
// ========= Sigmoid ===========
// =============================
TEST(SigmoidFun, ReturnsExpectedValues){
  NN::Sigmoid sigmoid;

  std::array<float, 3> layer = {0.0, 1.0, -1.0};

  EXPECT_FLOAT_EQ(sigmoid.fun(layer, 0), 0.5);
  EXPECT_NEAR(sigmoid.fun(layer, 1), 0.7310585786300049, 1e-6);
  EXPECT_NEAR(sigmoid.fun(layer, 2), 0.2689414213699951, 1e-6);
};

TEST(SigmoidFun, HandlesLargePositiveValue){
  NN::Sigmoid sigmoid;

  std::array<float, 1> layer = {1000.0};

  const float result = sigmoid.fun(layer, 0);

  EXPECT_TRUE(std::isfinite(result));
  EXPECT_FLOAT_EQ(result, 1.0);
};

TEST(SigmoidFun, HandlesLargeNegativeValue){
  NN::Sigmoid sigmoid;

  std::array<float, 1> layer = {-1000.0};

  const float result = sigmoid.fun(layer, 0);

  EXPECT_TRUE(std::isfinite(result));
  EXPECT_FLOAT_EQ(result, 0.0);
};

TEST(SigmoidFun, ResultRemainsInRange){
  NN::Sigmoid sigmoid;

  for(float value = -100.0; value <= 100.0; value += 0.5){
    std::array<float, 1> layer = {value};

    const float result = sigmoid.fun(layer, 0);

    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
  };
};

TEST(SigmoidFun, IsMonotonicallyIncreasing){
  NN::Sigmoid sigmoid;

  std::array<float, 1> layer = {-10.0};

  float previous = sigmoid.fun(layer, 0);

  for(float value = -9.9; value <= 10.0; value += 0.1){
    layer[0] = value;

    const float current = sigmoid.fun(layer, 0);

    EXPECT_GE(current, previous);

    previous = current;
  };
};

TEST(SigmoidFun, IsSymmetricAroundHalf){
  NN::Sigmoid sigmoid;

  std::array<float, 2> layer = {1.0, -1.0};

  EXPECT_NEAR(sigmoid.fun(layer, 0) + sigmoid.fun(layer, 1), 1.0, 1e-6);

  layer = {5.0, -5.0};

  EXPECT_NEAR(sigmoid.fun(layer, 0) + sigmoid.fun(layer, 1), 1.0, 1e-6);

  layer = {10.0, -10.0};

  EXPECT_NEAR(sigmoid.fun(layer, 0) + sigmoid.fun(layer, 1), 1.0, 1e-6);
};

TEST(SigmoidFun, HandlesInfinity){
  NN::Sigmoid sigmoid;

  std::array<float, 2> layer = {
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()
  };

  EXPECT_FLOAT_EQ(sigmoid.fun(layer, 0), 1.0);
  EXPECT_FLOAT_EQ(sigmoid.fun(layer, 1), 0.0);
};

TEST(SigmoidFunPrime, ReturnsExpectedValues){
  NN::Sigmoid sigmoid;

  std::array<float, 3> layer = {0.0, 1.0, -1.0};

  EXPECT_FLOAT_EQ(sigmoid.fun_prime(layer, 0, 0), 0.25);
  EXPECT_NEAR(sigmoid.fun_prime(layer, 1, 1), 0.19661193324148185, 1e-6);
  EXPECT_NEAR(sigmoid.fun_prime(layer, 2, 2), 0.19661193324148185, 1e-6);
};

TEST(SigmoidFunPrime, ReturnsZeroOutsideDiagonal){
  NN::Sigmoid sigmoid;

  std::array<float, 3> layer = {0.0, 1.0, -1.0};

  EXPECT_FLOAT_EQ(sigmoid.fun_prime(layer, 0, 1), 0.0);
  EXPECT_FLOAT_EQ(sigmoid.fun_prime(layer, 0, 2), 0.0);
  EXPECT_FLOAT_EQ(sigmoid.fun_prime(layer, 1, 0), 0.0);
  EXPECT_FLOAT_EQ(sigmoid.fun_prime(layer, 2, 1), 0.0);
};

TEST(SigmoidFunPrime, ApproachesZeroForExtremeValues){
  NN::Sigmoid sigmoid;

  std::array<float, 2> layer = {1000.0, -1000.0};

  EXPECT_NEAR(sigmoid.fun_prime(layer, 0, 0), 0.0, 1e-6);
  EXPECT_NEAR(sigmoid.fun_prime(layer, 1, 1), 0.0, 1e-6);
};

TEST(SigmoidFunPrime, MaximumAtZero){
  NN::Sigmoid sigmoid;

  std::array<float, 1> layer = {0.0};

  const float zero = sigmoid.fun_prime(layer, 0, 0);

  for(float value = -10.0; value <= 10.0; value += 0.1){
    layer[0] = value;

    EXPECT_LE(sigmoid.fun_prime(layer, 0, 0), zero);
  };
};

TEST(SigmoidFunPrime, ResultRemainsInRange){
  NN::Sigmoid sigmoid;

  std::array<float, 1> layer;

  for(float value = -100.0; value <= 100.0; value += 0.5){
    layer[0] = value;

    const float result = sigmoid.fun_prime(layer, 0, 0);

    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 0.25);
  };
};

TEST(SigmoidFun, PropagatesNaN){
  NN::Sigmoid sigmoid;

  std::array<float, 1> layer = {std::numeric_limits<float>::quiet_NaN()};

  EXPECT_TRUE(std::isnan(sigmoid.fun(layer, 0)));
};

TEST(SigmoidFunPrime, IsSymmetric){
  NN::Sigmoid sigmoid;

  std::array<float, 2> layer = {3.0f, -3.0f};

  EXPECT_NEAR(sigmoid.fun_prime(layer, 0, 0), sigmoid.fun_prime(layer, 1, 1), 1e-6);
};

TEST(SigmoidFunPrime, IsZeroOutsideDiagonalForAllInputs){
  NN::Sigmoid sigmoid;

  std::array<float, 4> layer = {-10.0f, -1.0f, 1.0f, 10.0f};

  for(size_t i = 0; i < layer.size(); i++){
    for(size_t j = 0; j < layer.size(); j++){
      if(i != j) EXPECT_FLOAT_EQ(sigmoid.fun_prime(layer, i, j), 0.0f);
    };
  };
};



// =============================
// ========= Softmax ===========
// =============================
TEST(SoftmaxFun, ProbabilitiesSumToOne){
  NN::Softmax softmax;

  std::array<float, 3> layer = {1.0, 2.0, 3.0};

  float sum = 0.0;

  for(size_t i = 0; i < layer.size(); i++)
    sum += softmax.fun(layer, i);

  EXPECT_NEAR(sum, 1.0, 1e-6);
};

TEST(SoftmaxFun, ReturnsPositiveProbabilities){
  NN::Softmax softmax;

  std::array<float, 4> layer = {-10.0, -1.0, 0.0, 10.0};

  for(size_t i = 0; i < layer.size(); i++){
    EXPECT_GT(softmax.fun(layer, i), 0.0);
    EXPECT_LE(softmax.fun(layer, i), 1.0);
  };
};

TEST(SoftmaxFun, PreservesOrdering){
  NN::Softmax softmax;

  std::array<float, 3> layer = {1.0, 2.0, 3.0};

  EXPECT_LT(softmax.fun(layer, 0), softmax.fun(layer, 1));
  EXPECT_LT(softmax.fun(layer, 1), softmax.fun(layer, 2));
};

TEST(SoftmaxFun, UniformForEqualInputs){
  NN::Softmax softmax;

  std::array<float, 4> layer = {1.0, 1.0, 1.0, 1.0};

  for(size_t i = 0; i < layer.size(); i++)
    EXPECT_NEAR(softmax.fun(layer, i), 0.25, 1e-6);
};

TEST(SoftmaxFun, IsShiftInvariant){
  NN::Softmax softmax;

  std::array<float, 3> first = {1.0, 2.0, 3.0};
  std::array<float, 3> second = {101.0, 102.0, 103.0};

  for(size_t i = 0; i < first.size(); i++)
    EXPECT_NEAR(softmax.fun(first, i), softmax.fun(second, i), 1e-6);
};

TEST(SoftmaxFun, HandlesLargeValues){
  NN::Softmax softmax;

  std::array<float, 3> layer = {1000.0, 1001.0, 1002.0};

  float sum = 0.0;

  for(size_t i = 0; i < layer.size(); i++){
    const float result = softmax.fun(layer, i);

    EXPECT_TRUE(std::isfinite(result));

    sum += result;
  };

  EXPECT_NEAR(sum, 1.0, 1e-6);
};

TEST(SoftmaxFunPrime, DiagonalIsPositive){
  NN::Softmax softmax;

  std::array<float, 3> layer = {1.0, 2.0, 3.0};

  for(size_t i = 0; i < layer.size(); i++)
    EXPECT_GT(softmax.fun_prime(layer, i, i), 0.0);
};

TEST(SoftmaxFunPrime, OffDiagonalIsNegative){
  NN::Softmax softmax;

  std::array<float, 3> layer = {1.0, 2.0, 3.0};

  EXPECT_LT(softmax.fun_prime(layer, 0, 1), 0.0);
  EXPECT_LT(softmax.fun_prime(layer, 0, 2), 0.0);
  EXPECT_LT(softmax.fun_prime(layer, 1, 0), 0.0);
  EXPECT_LT(softmax.fun_prime(layer, 2, 1), 0.0);
};

TEST(SoftmaxFunPrime, JacobianRowsSumToZero){
  NN::Softmax softmax;

  std::array<float, 3> layer = {1.0, 2.0, 3.0};

  for(size_t i = 0; i < layer.size(); i++){
    float sum = 0.0;

    for(size_t j = 0; j < layer.size(); j++)
      sum += softmax.fun_prime(layer, i, j);

    EXPECT_NEAR(sum, 0.0, 1e-6);
  };
};

TEST(SoftmaxFun, SingleElementIsOne){
  NN::Softmax softmax;

  std::array<float, 1> layer = {123.0f};

  EXPECT_FLOAT_EQ(softmax.fun(layer, 0), 1.0f);
  EXPECT_FLOAT_EQ(softmax.fun_prime(layer, 0, 0), 0.0f);
};

TEST(SoftmaxFun, HandlesLargeNegativeValues){
  NN::Softmax softmax;

  std::array<float, 3> layer = {-1000.0f, -999.0f, -998.0f};

  float sum = 0.0f;

  for(size_t i = 0; i < layer.size(); i++){
    float result = softmax.fun(layer, i);

    EXPECT_TRUE(std::isfinite(result));
    EXPECT_GE(result, 0.0f);
    EXPECT_LE(result, 1.0f);

    sum += result;
  };

  EXPECT_NEAR(sum, 1.0f, 1e-6);
};

TEST(SoftmaxFun, HandlesEqualLargeValues){
  NN::Softmax softmax;

  std::array<float, 3> layer = {1000.0f, 1000.0f, 1000.0f};

  for(size_t i = 0; i < layer.size(); i++)
    EXPECT_NEAR(softmax.fun(layer, i), 1.0f / 3.0f, 1e-6);
};

TEST(SoftmaxFun, HandlesUnderflowWithoutNaN){
  NN::Softmax softmax;

  std::array<float, 3> layer = {-1000.0f, 0.0f, 1000.0f};

  float sum = 0.0f;

  for(size_t i = 0; i < layer.size(); i++){
    float result = softmax.fun(layer, i);

    EXPECT_TRUE(std::isfinite(result));
    EXPECT_GE(result, 0.0f);
    EXPECT_LE(result, 1.0f);

    sum += result;
  };

  EXPECT_NEAR(sum, 1.0f, 1e-6);
  EXPECT_NEAR(softmax.fun(layer, 2), 1.0f, 1e-6);
};

TEST(SoftmaxFun, IsShiftInvariantForNegativeShift){
  NN::Softmax softmax;

  std::array<float, 3> first = {1.0f, 2.0f, 3.0f};
  std::array<float, 3> second = {-999.0f, -998.0f, -997.0f};

  for(size_t i = 0; i < first.size(); i++)
    EXPECT_NEAR(softmax.fun(first, i), softmax.fun(second, i), 1e-6);
};

TEST(SoftmaxFunPrime, UniformJacobianHasExpectedValues){
  NN::Softmax softmax;

  std::array<float, 3> layer = {0.0f, 0.0f, 0.0f};

  for(size_t i = 0; i < layer.size(); i++){
    for(size_t j = 0; j < layer.size(); j++){
      float expected = i == j ? 2.0f / 9.0f : -1.0f / 9.0f;

      EXPECT_NEAR(softmax.fun_prime(layer, i, j), expected, 1e-6);
    };
  };
};

TEST(SoftmaxFunPrime, JacobianIsSymmetric){
  NN::Softmax softmax;

  std::array<float, 4> layer = {-2.0f, -0.5f, 1.0f, 3.0f};

  for(size_t i = 0; i < layer.size(); i++){
    for(size_t j = 0; j < layer.size(); j++)
      EXPECT_NEAR(softmax.fun_prime(layer, i, j), softmax.fun_prime(layer, j, i), 1e-6);
  };
};

TEST(SoftmaxFunPrime, JacobianColumnsSumToZero){
  NN::Softmax softmax;

  std::array<float, 4> layer = {-2.0f, -0.5f, 1.0f, 3.0f};

  for(size_t j = 0; j < layer.size(); j++){
    float sum = 0.0f;

    for(size_t i = 0; i < layer.size(); i++)
      sum += softmax.fun_prime(layer, i, j);

    EXPECT_NEAR(sum, 0.0f, 1e-6);
  };
};



// =============================
// ========== ReLU =============
// =============================
TEST(ReLUFun, ReturnsExpectedValues){
  NN::ReLU relu;

  std::array<float, 5> layer = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};

  EXPECT_FLOAT_EQ(relu.fun(layer, 0), 0.0f);
  EXPECT_FLOAT_EQ(relu.fun(layer, 1), 0.0f);
  EXPECT_FLOAT_EQ(relu.fun(layer, 2), 0.0f);
  EXPECT_FLOAT_EQ(relu.fun(layer, 3), 1.0f);
  EXPECT_FLOAT_EQ(relu.fun(layer, 4), 10.0f);
};

TEST(ReLUFun, HandlesExtremeValues){
  NN::ReLU relu;

  std::array<float, 4> layer = {
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::max(),
    -std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::infinity()
  };

  EXPECT_FLOAT_EQ(relu.fun(layer, 0), 0.0f);
  EXPECT_FLOAT_EQ(relu.fun(layer, 1), std::numeric_limits<float>::max());
  EXPECT_FLOAT_EQ(relu.fun(layer, 2), 0.0f);
  EXPECT_FLOAT_EQ(relu.fun(layer, 3), std::numeric_limits<float>::infinity());
};

TEST(ReLUFun, IsMonotonicallyIncreasing){
  NN::ReLU relu;

  std::array<float, 1> layer = {-10.0f};

  float previous = relu.fun(layer, 0);

  for(float value = -9.5f; value <= 10.0f; value += 0.5f){
    layer[0] = value;

    float current = relu.fun(layer, 0);

    EXPECT_GE(current, previous);

    previous = current;
  };
};

TEST(ReLUFunPrime, ReturnsExpectedValues){
  NN::ReLU relu;

  std::array<float, 3> layer = {-1.0f, 0.0f, 1.0f};

  EXPECT_FLOAT_EQ(relu.fun_prime(layer, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(relu.fun_prime(layer, 1, 1), 0.0f);
  EXPECT_FLOAT_EQ(relu.fun_prime(layer, 2, 2), 1.0f);
};

TEST(ReLUFunPrime, ReturnsZeroOutsideDiagonal){
  NN::ReLU relu;

  std::array<float, 3> layer = {-1.0f, 0.0f, 1.0f};

  for(size_t i = 0; i < layer.size(); i++){
    for(size_t j = 0; j < layer.size(); j++){
      if(i != j) EXPECT_FLOAT_EQ(relu.fun_prime(layer, i, j), 0.0f);
    };
  };
};

TEST(ReLUFunPrime, HandlesExtremeValues){
  NN::ReLU relu;

  std::array<float, 2> layer = {
    -std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::infinity()
  };

  EXPECT_FLOAT_EQ(relu.fun_prime(layer, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(relu.fun_prime(layer, 1, 1), 1.0f);
};

TEST(ActivationInterface, ReLUWorksPolymorphically){
  std::unique_ptr<NN::iActivation> activation = std::make_unique<NN::ReLU>();

  std::array<float, 2> layer = {-1.0f, 2.0f};

  EXPECT_FLOAT_EQ(activation->fun(layer, 0), 0.0f);
  EXPECT_FLOAT_EQ(activation->fun(layer, 1), 2.0f);
  EXPECT_FLOAT_EQ(activation->fun_prime(layer, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(activation->fun_prime(layer, 1, 1), 1.0f);
  EXPECT_FLOAT_EQ(activation->fun_prime(layer, 0, 1), 0.0f);
};

TEST(ReLUFun, HandlesSignedZero){
  NN::ReLU relu;

  std::array<float, 2> layer = {0.0f, -0.0f};

  EXPECT_FLOAT_EQ(relu.fun(layer, 0), 0.0f);
  EXPECT_FLOAT_EQ(relu.fun(layer, 1), 0.0f);
};

TEST(ReLUFun, HandlesNaN){
  NN::ReLU relu;

  std::array<float, 1> layer = {std::numeric_limits<float>::quiet_NaN()};

  EXPECT_FLOAT_EQ(relu.fun(layer, 0), 0.0f);
};

TEST(ReLUFunPrime, UsesZeroDerivativeAtZero){
  NN::ReLU relu;

  std::array<float, 2> layer = {0.0f, -0.0f};

  EXPECT_FLOAT_EQ(relu.fun_prime(layer, 0, 0), 0.0f);
  EXPECT_FLOAT_EQ(relu.fun_prime(layer, 1, 1), 0.0f);
};



// =============================
// ======= Mathematics =========
// =============================
template<typename Activation, size_t N>
void checkActivationJacobian(const std::array<float, N>& input){
  Activation activation;

  constexpr float epsilon = 0.001f;
  constexpr double tolerance = 0.0005;

  for(size_t i = 0; i < N; i++){
    for(size_t j = 0; j < N; j++){
      auto plus = input;
      auto minus = input;

      plus[j] += epsilon;
      minus[j] -= epsilon;

      double positive = activation.fun(plus, i);
      double negative = activation.fun(minus, i);
      double numerical = (positive - negative) / (2.0 * epsilon);
      double analytical = activation.fun_prime(input, i, j);

      SCOPED_TRACE(::testing::Message() << "i=" << i << " j=" << j);

      ASSERT_TRUE(std::isfinite(numerical));
      ASSERT_TRUE(std::isfinite(analytical));
      EXPECT_NEAR(analytical, numerical, tolerance);
    };
  };
};

TEST(LinearMathematics, NumericalJacobian){
  std::array<float, 4> layer = {-1.25f, -0.4f, 0.7f, 1.3f};

  checkActivationJacobian<NN::Linear>(layer);
};

TEST(SigmoidMathematics, NumericalJacobian){
  std::array<float, 4> layer = {-1.25f, -0.4f, 0.7f, 1.3f};

  checkActivationJacobian<NN::Sigmoid>(layer);
};

TEST(SoftmaxMathematics, NumericalJacobian){
  std::array<float, 4> layer = {-1.25f, -0.4f, 0.7f, 1.3f};

  checkActivationJacobian<NN::Softmax>(layer);
};

TEST(ReLUMathematics, NumericalJacobian){
  std::array<float, 4> layer = {-1.25f, -0.4f, 0.7f, 1.3f};

  checkActivationJacobian<NN::ReLU>(layer);
};



// =============================================================================
// ================================= Shaders ===================================
// =============================================================================



#include "NeuralNetwork/GPUAcceleration/GPUAcceleration.h"
#include "CWindow/Renderer/OpenGL/Shader/SSBO/Shader.h"



struct ActivationShaderBuffers {
  std::vector<CW::Renderer::GPUStore> storage;

  ActivationShaderBuffers(const std::vector<float>& nodes)
    : storage(7) {
    NN::GPUAcceleration::get();
    storage[2].set(nodes);
    storage[3].set(std::vector<float>(nodes.size(), 0.0f));
    storage[4].set(std::vector<float>(nodes.size(), 0.0f));
    storage[5].set(std::vector<float>(nodes.size(), 0.0f));
  };

  void run(CW::Renderer::ComputeShader& shader, unsigned int groups){
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
// ======= Linear Shader =======
// =============================
TEST(LinearShader, ReturnsInput){
  std::vector<float> input = {0.0f, 1.0f, -1.0f, 123.456f, -123.456f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationLinearShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++)
    EXPECT_FLOAT_EQ(result[i], input[i]);
};

TEST(LinearShader, HandlesExtremeValues){
  std::vector<float> input = {
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::lowest()
  };

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationLinearShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], std::numeric_limits<float>::max());
  EXPECT_FLOAT_EQ(result[1], std::numeric_limits<float>::lowest());
};

TEST(LinearShader, PropagatesNaN){
  std::vector<float> input = {std::numeric_limits<float>::quiet_NaN()};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationLinearShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(std::isnan(result[0]));
};

TEST(LinearShader, HandlesMultipleWorkgroups){
  std::vector<float> input(513);

  for(size_t i = 0; i < input.size(); i++)
    input[i] = float(i) * 0.25f - 64.0f;

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationLinearShader(), (input.size() + 255) / 256);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++)
    EXPECT_FLOAT_EQ(result[i], input[i]);
};



// =============================
// ===== Linear Prime Shader ===
// =============================
TEST(LinearPrimeShader, ReturnsOnes){
  std::vector<float> input = {0.0f, 1.0f, -1.0f, 123.456f, -123.456f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationLinearPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++)
    EXPECT_FLOAT_EQ(result[i], 1.0f);
};

TEST(LinearPrimeShader, IsIndependentOfInput){
  std::vector<float> input = {
    std::numeric_limits<float>::lowest(),
    0.0f,
    std::numeric_limits<float>::max()
  };

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationLinearPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++)
    EXPECT_FLOAT_EQ(result[i], 1.0f);
};

TEST(LinearPrimeShader, HandlesMultipleWorkgroups){
  std::vector<float> input(513, 2.0f);

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationLinearPrimeShader(), (input.size() + 255) / 256);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++)
    EXPECT_FLOAT_EQ(result[i], 1.0f);
};



// =============================
// ======= Sigmoid Shader ======
// =============================
TEST(SigmoidShader, ReturnsExpectedValues){
  std::vector<float> input = {0.0f, 1.0f, -1.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 0.5f);
  EXPECT_NEAR(result[1], 0.7310585786300049, 1e-6);
  EXPECT_NEAR(result[2], 0.2689414213699951, 1e-6);
};

TEST(SigmoidShader, HandlesLargePositiveValue){
  std::vector<float> input = {1000.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), 1);

  EXPECT_TRUE(std::isfinite(result[0]));
  EXPECT_FLOAT_EQ(result[0], 1.0f);
};

TEST(SigmoidShader, HandlesLargeNegativeValue){
  std::vector<float> input = {-1000.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), 1);

  EXPECT_TRUE(std::isfinite(result[0]));
  EXPECT_FLOAT_EQ(result[0], 0.0f);
};

TEST(SigmoidShader, ResultRemainsInRange){
  std::vector<float> input(401);

  for(size_t i = 0; i < input.size(); i++)
    input[i] = -100.0f + float(i) * 0.5f;

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), (input.size() + 255) / 256);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++){
    EXPECT_TRUE(std::isfinite(result[i]));
    EXPECT_GE(result[i], 0.0f);
    EXPECT_LE(result[i], 1.0f);
  };
};

TEST(SigmoidShader, IsMonotonicallyIncreasing){
  std::vector<float> input(201);

  for(size_t i = 0; i < input.size(); i++)
    input[i] = -10.0f + float(i) * 0.1f;

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 1; i < result.size(); i++)
    EXPECT_GE(result[i], result[i - 1]);
};

TEST(SigmoidShader, IsSymmetricAroundHalf){
  std::vector<float> input = {1.0f, -1.0f, 5.0f, -5.0f, 10.0f, -10.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i += 2)
    EXPECT_NEAR(result[i] + result[i + 1], 1.0f, 1e-6);
};

TEST(SigmoidShader, HandlesInfinity){
  std::vector<float> input = {
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()
  };

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 1.0f);
  EXPECT_FLOAT_EQ(result[1], 0.0f);
};

TEST(SigmoidShader, HandlesMultipleWorkgroups){
  std::vector<float> input(513);

  for(size_t i = 0; i < input.size(); i++)
    input[i] = float(int(i % 21) - 10) * 0.5f;

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), (input.size() + 255) / 256);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++){
    double expected = 1.0 / (1.0 + std::exp(-double(input[i])));

    SCOPED_TRACE(i);
    EXPECT_NEAR(result[i], expected, 1e-6);
  };
};



// =============================
// ==== Sigmoid Prime Shader ===
// =============================
TEST(SigmoidPrimeShader, ReturnsExpectedValues){
  std::vector<float> input = {0.0f, 1.0f, -1.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), 1);
  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 0.25f);
  EXPECT_NEAR(result[1], 0.19661193324148185, 1e-6);
  EXPECT_NEAR(result[2], 0.19661193324148185, 1e-6);
};

TEST(SigmoidPrimeShader, ApproachesZeroForExtremeValues){
  std::vector<float> input = {1000.0f, -1000.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), 1);
  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_TRUE(std::isfinite(result[0]));
  EXPECT_TRUE(std::isfinite(result[1]));
  EXPECT_NEAR(result[0], 0.0f, 1e-6);
  EXPECT_NEAR(result[1], 0.0f, 1e-6);
};

TEST(SigmoidPrimeShader, MaximumAtZero){
  std::vector<float> input(201);

  for(size_t i = 0; i < input.size(); i++)
    input[i] = -10.0f + float(i) * 0.1f;

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), 1);
  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < result.size(); i++)
    EXPECT_LE(result[i], 0.25f + 1e-7f);
};

TEST(SigmoidPrimeShader, ResultRemainsInRange){
  std::vector<float> input(401);

  for(size_t i = 0; i < input.size(); i++)
    input[i] = -100.0f + float(i) * 0.5f;

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), (input.size() + 255) / 256);
  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidPrimeShader(), (input.size() + 255) / 256);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < result.size(); i++){
    EXPECT_TRUE(std::isfinite(result[i]));
    EXPECT_GE(result[i], 0.0f);
    EXPECT_LE(result[i], 0.25f + 1e-7f);
  };
};

TEST(SigmoidPrimeShader, NumericalDerivative){
  std::vector<float> input = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), 1);
  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++){
    double x = input[i];
    double positive = 1.0 / (1.0 + std::exp(-(x + 0.001)));
    double negative = 1.0 / (1.0 + std::exp(-(x - 0.001)));
    double numerical = (positive - negative) / 0.002;

    SCOPED_TRACE(i);
    EXPECT_NEAR(result[i], numerical, 1e-5);
  };
};

TEST(SigmoidPrimeShader, HandlesMultipleWorkgroups){
  std::vector<float> input(513);

  for(size_t i = 0; i < input.size(); i++)
    input[i] = float(int(i % 21) - 10) * 0.5f;

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidShader(), (input.size() + 255) / 256);
  buffers.run(NN::GPUAcceleration::get().getActivationSigmoidPrimeShader(), (input.size() + 255) / 256);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++){
    double y = 1.0 / (1.0 + std::exp(-double(input[i])));
    double expected = y * (1.0 - y);

    SCOPED_TRACE(i);
    EXPECT_NEAR(result[i], expected, 1e-6);
  };
};



// =============================
// ======== ReLU Shader ========
// =============================
TEST(ReLUShader, ReturnsExpectedValues){
  std::vector<float> input = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationReLUShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], 0.0f);
  EXPECT_FLOAT_EQ(result[2], 0.0f);
  EXPECT_FLOAT_EQ(result[3], 1.0f);
  EXPECT_FLOAT_EQ(result[4], 10.0f);
};

TEST(ReLUShader, HandlesExtremeValues){
  std::vector<float> input = {
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::max(),
    -std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::infinity()
  };

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationReLUShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], std::numeric_limits<float>::max());
  EXPECT_FLOAT_EQ(result[2], 0.0f);
  EXPECT_FLOAT_EQ(result[3], std::numeric_limits<float>::infinity());
};

TEST(ReLUShader, HandlesSignedZero){
  std::vector<float> input = {0.0f, -0.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationReLUShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], 0.0f);
};

TEST(ReLUShader, IsMonotonicallyIncreasing){
  std::vector<float> input(41);

  for(size_t i = 0; i < input.size(); i++)
    input[i] = -10.0f + float(i) * 0.5f;

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationReLUShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 1; i < result.size(); i++)
    EXPECT_GE(result[i], result[i - 1]);
};

TEST(ReLUShader, HandlesMultipleWorkgroups){
  std::vector<float> input(513);

  for(size_t i = 0; i < input.size(); i++)
    input[i] = float(int(i % 21) - 10) * 0.5f;

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationReLUShader(), (input.size() + 255) / 256);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++){
    SCOPED_TRACE(i);
    EXPECT_FLOAT_EQ(result[i], std::max(input[i], 0.0f));
  };
};



// =============================
// ===== ReLU Prime Shader =====
// =============================
TEST(ReLUPrimeShader, ReturnsExpectedValues){
  std::vector<float> input = {-1.0f, 0.0f, 1.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationReLUPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], 0.0f);
  EXPECT_FLOAT_EQ(result[2], 1.0f);
};

TEST(ReLUPrimeShader, UsesZeroDerivativeAtZero){
  std::vector<float> input = {0.0f, -0.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationReLUPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], 0.0f);
};

TEST(ReLUPrimeShader, HandlesExtremeValues){
  std::vector<float> input = {
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::max(),
    -std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::infinity()
  };

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationReLUPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], 1.0f);
  EXPECT_FLOAT_EQ(result[2], 0.0f);
  EXPECT_FLOAT_EQ(result[3], 1.0f);
};

TEST(ReLUPrimeShader, NumericalDerivative){
  std::vector<float> input = {-3.0f, -1.0f, 1.0f, 3.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationReLUPrimeShader(), 1);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++){
    double x = input[i];
    double positive = std::max(0.0, x + 0.001);
    double negative = std::max(0.0, x - 0.001);
    double numerical = (positive - negative) / 0.002;

    SCOPED_TRACE(i);
    EXPECT_NEAR(result[i], numerical, 1e-6);
  };
};

TEST(ReLUPrimeShader, HandlesMultipleWorkgroups){
  std::vector<float> input(513);

  for(size_t i = 0; i < input.size(); i++)
    input[i] = float(int(i % 21) - 10) * 0.5f;

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationReLUPrimeShader(), (input.size() + 255) / 256);

  std::vector<float> result = buffers.get(5);

  ASSERT_EQ(result.size(), input.size());

  for(size_t i = 0; i < input.size(); i++){
    SCOPED_TRACE(i);
    EXPECT_FLOAT_EQ(result[i], input[i] > 0.0f ? 1.0f : 0.0f);
  };
};



// =============================
// ======= Softmax Shader ======
// =============================
TEST(SoftmaxShader, ProbabilitiesSumToOne){
  std::vector<float> input = {1.0f, 2.0f, 3.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  float sum = 0.0f;

  for(float value : result){
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_GE(value, 0.0f);
    EXPECT_LE(value, 1.0f);
    sum += value;
  };

  EXPECT_NEAR(sum, 1.0f, 1e-6);
};

TEST(SoftmaxShader, ReturnsExpectedValues){
  std::vector<float> input = {1.0f, 2.0f, 3.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_NEAR(result[0], 0.09003057317038046, 1e-6);
  EXPECT_NEAR(result[1], 0.24472847105479764, 1e-6);
  EXPECT_NEAR(result[2], 0.6652409557748218, 1e-6);
};

TEST(SoftmaxShader, PreservesOrdering){
  std::vector<float> input = {1.0f, 2.0f, 3.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  EXPECT_LT(result[0], result[1]);
  EXPECT_LT(result[1], result[2]);
};

TEST(SoftmaxShader, UniformForEqualInputs){
  std::vector<float> input = {1.0f, 1.0f, 1.0f, 1.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  for(float value : result)
    EXPECT_NEAR(value, 0.25f, 1e-6);
};

TEST(SoftmaxShader, IsShiftInvariant){
  std::vector<float> first = {1.0f, 2.0f, 3.0f};
  std::vector<float> second = {101.0f, 102.0f, 103.0f};

  ActivationShaderBuffers first_buffers(first);
  ActivationShaderBuffers second_buffers(second);

  first_buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);
  second_buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);

  std::vector<float> first_result = first_buffers.get(3);
  std::vector<float> second_result = second_buffers.get(3);

  ASSERT_EQ(first_result.size(), second_result.size());

  for(size_t i = 0; i < first_result.size(); i++)
    EXPECT_NEAR(first_result[i], second_result[i], 1e-6);
};

TEST(SoftmaxShader, HandlesLargeValues){
  std::vector<float> input = {1000.0f, 1001.0f, 1002.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  float sum = 0.0f;

  for(float value : result){
    EXPECT_TRUE(std::isfinite(value));
    sum += value;
  };

  EXPECT_NEAR(sum, 1.0f, 1e-6);
};

TEST(SoftmaxShader, HandlesLargeNegativeValues){
  std::vector<float> input = {-1000.0f, -999.0f, -998.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  float sum = 0.0f;

  for(float value : result){
    EXPECT_TRUE(std::isfinite(value));
    sum += value;
  };

  EXPECT_NEAR(sum, 1.0f, 1e-6);
};

TEST(SoftmaxShader, HandlesUnderflowWithoutNaN){
  std::vector<float> input = {-1000.0f, 0.0f, 1000.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), input.size());

  float sum = 0.0f;

  for(float value : result){
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_GE(value, 0.0f);
    EXPECT_LE(value, 1.0f);
    sum += value;
  };

  EXPECT_NEAR(sum, 1.0f, 1e-6);
  EXPECT_NEAR(result[2], 1.0f, 1e-6);
};

TEST(SoftmaxShader, SingleElementIsOne){
  std::vector<float> input = {123.0f};

  ActivationShaderBuffers buffers(input);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);

  std::vector<float> result = buffers.get(3);

  ASSERT_EQ(result.size(), 1);

  EXPECT_FLOAT_EQ(result[0], 1.0f);
};

// TEST(SoftmaxShader, HandlesMoreThanOneWorkgroupOfValues){
//   std::vector<float> input(513);

//   for(size_t i = 0; i < input.size(); i++)
//     input[i] = float(int(i % 17) - 8) * 0.25f;

//   ActivationShaderBuffers buffers(input);

//   buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);

//   std::vector<float> result = buffers.get(3);

//   ASSERT_EQ(result.size(), input.size());

//   double max = *std::max_element(input.begin(), input.end());
//   double sum = 0.0;

//   for(float value : input)
//     sum += std::exp(double(value) - max);

//   double probability_sum = 0.0;

//   for(size_t i = 0; i < input.size(); i++){
//     double expected = std::exp(double(input[i]) - max) / sum;

//     SCOPED_TRACE(i);
//     EXPECT_NEAR(result[i], expected, 1e-6);
//     probability_sum += result[i];
//   };

//   EXPECT_NEAR(probability_sum, 1.0, 1e-5);
// };



// =============================
// ==== Softmax Prime Shader ===
// =============================
TEST(SoftmaxPrimeShader, ReturnsExpectedValues){
  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> gradient = {0.2f, -0.3f, 0.5f};

  ActivationShaderBuffers buffers(input);

  buffers.storage[5].set(gradient);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);
  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxPrimeShader(), 1);

  std::vector<float> result = buffers.get(4);

  ASSERT_EQ(result.size(), input.size());

  std::vector<float> probabilities = buffers.get(3);
  double dot = 0.0;

  for(size_t i = 0; i < input.size(); i++)
    dot += double(probabilities[i]) * gradient[i];

  for(size_t i = 0; i < input.size(); i++){
    double expected = double(probabilities[i]) * (gradient[i] - dot);

    SCOPED_TRACE(i);
    EXPECT_NEAR(result[i], expected, 1e-6);
  };
};

TEST(SoftmaxPrimeShader, UniformGradientProducesZero){
  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> gradient = {1.0f, 1.0f, 1.0f};

  ActivationShaderBuffers buffers(input);

  buffers.storage[5].set(gradient);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);
  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxPrimeShader(), 1);

  std::vector<float> result = buffers.get(4);

  ASSERT_EQ(result.size(), input.size());

  for(float value : result)
    EXPECT_NEAR(value, 0.0f, 1e-6);
};

TEST(SoftmaxPrimeShader, SigmaSumsToZero){
  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> gradient = {0.2f, -0.3f, 0.5f};

  ActivationShaderBuffers buffers(input);

  buffers.storage[5].set(gradient);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);
  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxPrimeShader(), 1);

  std::vector<float> result = buffers.get(4);

  ASSERT_EQ(result.size(), input.size());

  float sum = 0.0f;

  for(float value : result)
    sum += value;

  EXPECT_NEAR(sum, 0.0f, 1e-6);
};

TEST(SoftmaxPrimeShader, SingleElementIsZero){
  std::vector<float> input = {123.0f};
  std::vector<float> gradient = {2.0f};

  ActivationShaderBuffers buffers(input);

  buffers.storage[5].set(gradient);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);
  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxPrimeShader(), 1);

  std::vector<float> result = buffers.get(4);

  ASSERT_EQ(result.size(), 1);

  EXPECT_FLOAT_EQ(result[0], 0.0f);
};

TEST(SoftmaxPrimeShader, NumericalJacobianVectorProduct){
  std::vector<float> input = {-1.25f, -0.4f, 0.7f, 1.3f};
  std::vector<float> gradient = {0.2f, -0.3f, 0.5f, -0.1f};

  ActivationShaderBuffers buffers(input);

  buffers.storage[5].set(gradient);

  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);
  buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxPrimeShader(), 1);

  std::vector<float> result = buffers.get(4);

  ASSERT_EQ(result.size(), input.size());

  NN::Softmax softmax;

  for(size_t j = 0; j < input.size(); j++){
    auto plus = input;
    auto minus = input;

    plus[j] += 0.001f;
    minus[j] -= 0.001f;

    double positive = 0.0;
    double negative = 0.0;

    for(size_t i = 0; i < input.size(); i++){
      positive += double(gradient[i]) * softmax.fun(plus, i);
      negative += double(gradient[i]) * softmax.fun(minus, i);
    };

    double numerical = (positive - negative) / double(plus[j] - minus[j]);

    SCOPED_TRACE(j);
    EXPECT_NEAR(result[j], numerical, 0.0005);
  };
};

// TEST(SoftmaxPrimeShader, HandlesMoreThanOneWorkgroupOfValues){
//   std::vector<float> input(513);
//   std::vector<float> gradient(513);

//   for(size_t i = 0; i < input.size(); i++){
//     input[i] = float(int(i % 17) - 8) * 0.25f;
//     gradient[i] = float(int(i % 11) - 5) * 0.1f;
//   };

//   ActivationShaderBuffers buffers(input);

//   buffers.storage[5].set(gradient);

//   buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxShader(), 1);
//   buffers.run(NN::GPUAcceleration::get().getActivationSoftmaxPrimeShader(), 1);

//   std::vector<float> result = buffers.get(4);

//   ASSERT_EQ(result.size(), input.size());

//   double max = *std::max_element(input.begin(), input.end());
//   double sum = 0.0;

//   for(float value : input)
//     sum += std::exp(double(value) - max);

//   double dot = 0.0;

//   for(size_t i = 0; i < input.size(); i++)
//     dot += std::exp(double(input[i]) - max) / sum * gradient[i];

//   double sigma_sum = 0.0;

//   for(size_t i = 0; i < input.size(); i++){
//     double probability = std::exp(double(input[i]) - max) / sum;
//     double expected = probability * (gradient[i] - dot);

//     SCOPED_TRACE(i);
//     ASSERT_TRUE(std::isfinite(result[i]));
//     EXPECT_NEAR(result[i], expected, 1e-6);

//     sigma_sum += result[i];
//   };

//   EXPECT_NEAR(sigma_sum, 0.0, 1e-5);
// };



// =============================
// ===== Shader Mathematics ====
// =============================
void checkElementwiseShaderJacobian(CW::Renderer::ComputeShader& shader, CW::Renderer::ComputeShader& prime, const std::vector<float>& input){
  ActivationShaderBuffers buffers(input);

  const unsigned int groups = (input.size() + 255) / 256;

  buffers.run(shader, groups);
  buffers.run(prime, groups);

  std::vector<float> derivative = buffers.get(5);

  ASSERT_EQ(derivative.size(), input.size());

  constexpr float epsilon = 0.001f;
  constexpr double tolerance = 0.001;

  for(size_t j = 0; j < input.size(); j++){
    auto plus = input;
    auto minus = input;

    plus[j] += epsilon;
    minus[j] -= epsilon;

    buffers.storage[2].set(plus);
    buffers.run(shader, groups);
    std::vector<float> positive = buffers.get(3);

    buffers.storage[2].set(minus);
    buffers.run(shader, groups);
    std::vector<float> negative = buffers.get(3);

    ASSERT_EQ(positive.size(), input.size());
    ASSERT_EQ(negative.size(), input.size());

    for(size_t i = 0; i < input.size(); i++){
      double numerical = (double(positive[i]) - negative[i]) / double(plus[j] - minus[j]);
      double analytical = i == j ? derivative[i] : 0.0;

      SCOPED_TRACE(::testing::Message() << "i=" << i << " j=" << j);

      ASSERT_TRUE(std::isfinite(numerical));
      ASSERT_TRUE(std::isfinite(analytical));
      EXPECT_NEAR(analytical, numerical, tolerance);
    };
  };
};

TEST(LinearShaderMathematics, NumericalJacobian){
  std::vector<float> input = {-1.25f, -0.4f, 0.7f, 1.3f};

  auto& gpu = NN::GPUAcceleration::get();

  checkElementwiseShaderJacobian(gpu.getActivationLinearShader(), gpu.getActivationLinearPrimeShader(), input);
};

TEST(SigmoidShaderMathematics, NumericalJacobian){
  std::vector<float> input = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};

  auto& gpu = NN::GPUAcceleration::get();

  checkElementwiseShaderJacobian(gpu.getActivationSigmoidShader(), gpu.getActivationSigmoidPrimeShader(), input);
};

TEST(ReLUShaderMathematics, NumericalJacobian){
  std::vector<float> input = {-3.0f, -1.0f, 1.0f, 3.0f};

  auto& gpu = NN::GPUAcceleration::get();

  checkElementwiseShaderJacobian(gpu.getActivationReLUShader(), gpu.getActivationReLUPrimeShader(), input);
};

TEST(SoftmaxShaderMathematics, NumericalJacobian){
  std::vector<float> input = {-1.25f, -0.4f, 0.7f, 1.3f};

  ActivationShaderBuffers buffers(input);

  auto& gpu = NN::GPUAcceleration::get();

  constexpr float epsilon = 0.001f;
  constexpr double tolerance = 0.001;

  for(size_t j = 0; j < input.size(); j++){
    std::vector<float> gradient(input.size(), 0.0f);
    gradient[j] = 1.0f;

    buffers.storage[2].set(input);
    buffers.storage[5].set(gradient);

    buffers.run(gpu.getActivationSoftmaxShader(), 1);
    buffers.run(gpu.getActivationSoftmaxPrimeShader(), 1);

    std::vector<float> analytical = buffers.get(4);

    auto plus = input;
    auto minus = input;

    plus[j] += epsilon;
    minus[j] -= epsilon;

    buffers.storage[2].set(plus);
    buffers.run(gpu.getActivationSoftmaxShader(), 1);
    std::vector<float> positive = buffers.get(3);

    buffers.storage[2].set(minus);
    buffers.run(gpu.getActivationSoftmaxShader(), 1);
    std::vector<float> negative = buffers.get(3);

    ASSERT_EQ(analytical.size(), input.size());
    ASSERT_EQ(positive.size(), input.size());
    ASSERT_EQ(negative.size(), input.size());

    for(size_t i = 0; i < input.size(); i++){
      double numerical = (double(positive[i]) - negative[i]) / double(plus[j] - minus[j]);

      SCOPED_TRACE(::testing::Message() << "i=" << i << " j=" << j);

      ASSERT_TRUE(std::isfinite(numerical));
      ASSERT_TRUE(std::isfinite(analytical[i]));
      EXPECT_NEAR(analytical[i], numerical, tolerance);
    };
  };
};

TEST(SoftmaxShaderMathematics, CrossEntropyGradient){
  std::vector<float> input = {-1.25f, -0.4f, 0.7f, 1.3f};
  std::vector<float> target = {0.0f, 1.0f, 0.0f, 0.0f};

  ActivationShaderBuffers buffers(input);

  auto& gpu = NN::GPUAcceleration::get();

  buffers.run(gpu.getActivationSoftmaxShader(), 1);

  std::vector<float> probabilities = buffers.get(3);
  std::vector<float> gradient(input.size(), 0.0f);

  ASSERT_EQ(probabilities.size(), input.size());

  for(size_t i = 0; i < input.size(); i++){
    ASSERT_TRUE(std::isfinite(probabilities[i]));
    ASSERT_GT(probabilities[i], 0.0f);

    gradient[i] = -target[i] / probabilities[i];
  };

  buffers.storage[5].set(gradient);

  buffers.run(gpu.getActivationSoftmaxPrimeShader(), 1);

  std::vector<float> sigma = buffers.get(4);

  ASSERT_EQ(sigma.size(), input.size());

  for(size_t i = 0; i < input.size(); i++){
    SCOPED_TRACE(i);

    ASSERT_TRUE(std::isfinite(sigma[i]));
    EXPECT_NEAR(sigma[i], probabilities[i] - target[i], 1e-6);
  };

  auto error = [&](const std::vector<float>& nodes){
    buffers.storage[2].set(nodes);
    buffers.run(gpu.getActivationSoftmaxShader(), 1);

    std::vector<float> output = buffers.get(3);

    return -std::log(double(output[1]));
  };

  constexpr float epsilon = 0.001f;

  for(size_t j = 0; j < input.size(); j++){
    auto plus = input;
    auto minus = input;

    plus[j] += epsilon;
    minus[j] -= epsilon;

    double positive = error(plus);
    double negative = error(minus);
    double numerical = (positive - negative) / double(plus[j] - minus[j]);

    SCOPED_TRACE(j);

    ASSERT_TRUE(std::isfinite(numerical));
    EXPECT_NEAR(sigma[j], numerical, 0.001);
  };
};