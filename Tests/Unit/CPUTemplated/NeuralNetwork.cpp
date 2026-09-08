// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>
#include <tuple>
#include <utility>
#include <type_traits>
#include <cmath>
#include <memory>
#include <string>
#include <stdexcept>
#include <random>

#define private public
#define protected public

#include "NeuralNetwork/NeuralNetwork.h"

#undef private
#undef protected



namespace {
class TestLoss : public NN::iLoss{
public:
  float fun(float x, float t) const noexcept override{
    return std::abs(x - t);
  };

  float fun_prime(float x, float t) const noexcept override{
    if(x > t) return 1.0;
    if(x < t) return -1.0;
    return 0.0;
  };
};
};



// =============================
// ======== LayerBuilder =======
// =============================
TEST(LayerBuilder, BuildsSingleLayer){
  using Tuple = NN::LayerBuilder<2, 1>::type;

  EXPECT_EQ(std::tuple_size_v<Tuple>, 1u);
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<0, Tuple>, NN::Layer<2, 1>>));
};

TEST(LayerBuilder, BuildsTwoLayers){
  using Tuple = NN::LayerBuilder<2, 3, 1>::type;

  EXPECT_EQ(std::tuple_size_v<Tuple>, 2u);
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<0, Tuple>, NN::Layer<2, 3>>));
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<1, Tuple>, NN::Layer<3, 1>>));
};

TEST(LayerBuilder, BuildsMultipleLayers){
  using Tuple = NN::LayerBuilder<3, 8, 4, 2, 1>::type;

  EXPECT_EQ(std::tuple_size_v<Tuple>, 4u);
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<0, Tuple>, NN::Layer<3, 8>>));
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<1, Tuple>, NN::Layer<8, 4>>));
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<2, Tuple>, NN::Layer<4, 2>>));
  EXPECT_TRUE((std::is_same_v<std::tuple_element_t<3, Tuple>, NN::Layer<2, 1>>));
};



// =============================
// ===== Reverse Sequence ======
// =============================
TEST(ReverseSequence, HandlesEmptySequence){
  using Result = decltype(NN::reverse_sequence(std::make_index_sequence<0>{}));
  using Expected = std::index_sequence<>;

  EXPECT_TRUE((std::is_same_v<Result, Expected>));
};

TEST(ReverseSequence, ReversesSingleElement){
  using Result = decltype(NN::reverse_sequence(std::make_index_sequence<1>{}));
  using Expected = std::index_sequence<0>;

  EXPECT_TRUE((std::is_same_v<Result, Expected>));
};

TEST(ReverseSequence, ReversesTwoElements){
  using Result = decltype(NN::reverse_sequence(std::make_index_sequence<2>{}));
  using Expected = std::index_sequence<1, 0>;

  EXPECT_TRUE((std::is_same_v<Result, Expected>));
};

TEST(ReverseSequence, ReversesMultipleElements){
  using Result = decltype(NN::reverse_sequence(std::make_index_sequence<5>{}));
  using Expected = std::index_sequence<4, 3, 2, 1, 0>;

  EXPECT_TRUE((std::is_same_v<Result, Expected>));
};



// =============================
// ======= Constructors ========
// =============================
TEST(NeuralNetworkDefaultConstructor, InitializesLayers){
  NN::NeuralNetwork<2, 3, 1> network;

  EXPECT_EQ(std::tuple_size_v<decltype(network.layers)>, 2u);
};

TEST(NeuralNetworkDefaultConstructor, InitializesDefaultLearningRate){
  NN::NeuralNetwork<2, 3, 1> network;

  EXPECT_NEAR(std::get<0>(network.layers).getLearningRate(), 0.005, 1e-6);
  EXPECT_NEAR(std::get<1>(network.layers).getLearningRate(), 0.005, 1e-6);
};

TEST(NeuralNetworkDefaultConstructor, InitializesDefaultActivation){
  NN::NeuralNetwork<2, 3, 1> network;

  ASSERT_NE(std::get<0>(network.layers).getActivation(), nullptr);
  ASSERT_NE(std::get<1>(network.layers).getActivation(), nullptr);

  EXPECT_NE(dynamic_cast<NN::Linear*>(std::get<0>(network.layers).getActivation().get()), nullptr);
  EXPECT_NE(dynamic_cast<NN::Linear*>(std::get<1>(network.layers).getActivation().get()), nullptr);
};

TEST(NeuralNetworkDefaultConstructor, InitializesDefaultLoss){
  NN::NeuralNetwork<2, 3, 1> network;

  ASSERT_NE(std::get<0>(network.layers).getLoss(), nullptr);
  ASSERT_NE(std::get<1>(network.layers).getLoss(), nullptr);

  EXPECT_NE(dynamic_cast<NN::MSE*>(std::get<0>(network.layers).getLoss().get()), nullptr);
  EXPECT_NE(dynamic_cast<NN::MSE*>(std::get<1>(network.layers).getLoss().get()), nullptr);
};



// =============================
// ===== Learning Rate =========
// =============================
TEST(NeuralNetworkSetLearningRate, SetsSingleLayer){
  NN::NeuralNetwork<2, 1> network;

  network.setLearningRate(0.001);

  EXPECT_NEAR(std::get<0>(network.layers).getLearningRate(), 0.001, 1e-6);
};

TEST(NeuralNetworkSetLearningRate, SetsEveryLayer){
  NN::NeuralNetwork<2, 3, 2, 1> network;

  network.setLearningRate(0.001);

  EXPECT_NEAR(std::get<0>(network.layers).getLearningRate(), 0.001, 1e-6);
  EXPECT_NEAR(std::get<1>(network.layers).getLearningRate(), 0.001, 1e-6);
  EXPECT_NEAR(std::get<2>(network.layers).getLearningRate(), 0.001, 1e-6);
};

TEST(NeuralNetworkSetLearningRate, ReplacesPreviousLearningRate){
  NN::NeuralNetwork<2, 3, 1> network;

  network.setLearningRate(0.1);
  network.setLearningRate(0.0001);

  EXPECT_NEAR(std::get<0>(network.layers).getLearningRate(), 0.0001, 1e-6);
  EXPECT_NEAR(std::get<1>(network.layers).getLearningRate(), 0.0001, 1e-6);
};



// =============================
// ======== Activation =========
// =============================
TEST(NeuralNetworkSetActivation, SetsFirstLayer){
  NN::NeuralNetwork<2, 3, 1> network;

  network.setActivation<0, NN::Sigmoid>();

  EXPECT_NE(dynamic_cast<NN::Sigmoid*>(std::get<0>(network.layers).getActivation().get()), nullptr);
};

TEST(NeuralNetworkSetActivation, SetsRequestedLayerOnly){
  NN::NeuralNetwork<2, 3, 2, 1> network;

  network.setActivation<1, NN::Sigmoid>();

  EXPECT_NE(dynamic_cast<NN::Linear*>(std::get<0>(network.layers).getActivation().get()), nullptr);
  EXPECT_NE(dynamic_cast<NN::Sigmoid*>(std::get<1>(network.layers).getActivation().get()), nullptr);
  EXPECT_NE(dynamic_cast<NN::Linear*>(std::get<2>(network.layers).getActivation().get()), nullptr);
};

TEST(NeuralNetworkSetActivation, ReplacesPreviousActivation){
  NN::NeuralNetwork<2, 3, 1> network;

  network.setActivation<0, NN::Sigmoid>();
  network.setActivation<0, NN::Linear>();

  EXPECT_NE(dynamic_cast<NN::Linear*>(std::get<0>(network.layers).getActivation().get()), nullptr);
};



// =============================
// =========== Loss ============
// =============================
TEST(NeuralNetworkSetLoss, SetsRequestedLayer){
  NN::NeuralNetwork<2, 3, 1> network;

  network.setLoss<0, TestLoss>();

  EXPECT_NE(dynamic_cast<TestLoss*>(std::get<0>(network.layers).getLoss().get()), nullptr);
};

TEST(NeuralNetworkSetLoss, SetsRequestedLayerOnly){
  NN::NeuralNetwork<2, 3, 2, 1> network;

  network.setLoss<1, TestLoss>();

  EXPECT_NE(dynamic_cast<NN::MSE*>(std::get<0>(network.layers).getLoss().get()), nullptr);
  EXPECT_NE(dynamic_cast<TestLoss*>(std::get<1>(network.layers).getLoss().get()), nullptr);
  EXPECT_NE(dynamic_cast<NN::MSE*>(std::get<2>(network.layers).getLoss().get()), nullptr);
};

TEST(NeuralNetworkSetLoss, ReplacesPreviousLoss){
  NN::NeuralNetwork<2, 3, 1> network;

  network.setLoss<0, TestLoss>();
  network.setLoss<0, NN::MSE>();

  EXPECT_NE(dynamic_cast<NN::MSE*>(std::get<0>(network.layers).getLoss().get()), nullptr);
};



// =============================
// ========= setInput ==========
// =============================
TEST(NeuralNetworkSetInput, SetsInputLayerNodes){
  NN::NeuralNetwork<3, 2, 1> network;

  network.setInput({1.0, 2.0, 3.0});

  EXPECT_NEAR(std::get<0>(network.layers)[0], 1.0, 1e-6);
  EXPECT_NEAR(std::get<0>(network.layers)[1], 2.0, 1e-6);
  EXPECT_NEAR(std::get<0>(network.layers)[2], 3.0, 1e-6);
};

TEST(NeuralNetworkSetInput, DoesNotModifyOtherLayers){
  NN::NeuralNetwork<2, 2, 1> network;

  std::get<1>(network.layers).setNodes({8.0, 9.0});

  network.setInput({1.0, 2.0});

  EXPECT_NEAR(std::get<0>(network.layers)[0], 1.0, 1e-6);
  EXPECT_NEAR(std::get<0>(network.layers)[1], 2.0, 1e-6);

  EXPECT_NEAR(std::get<1>(network.layers)[0], 8.0, 1e-6);
  EXPECT_NEAR(std::get<1>(network.layers)[1], 9.0, 1e-6);
};

TEST(NeuralNetworkSetInput, IgnoresAdditionalNodes){
  NN::NeuralNetwork<2, 2, 1> network;

  network.setInput({1.0, 2.0, 3.0, 4.0});

  EXPECT_NEAR(std::get<0>(network.layers)[0], 1.0, 1e-6);
  EXPECT_NEAR(std::get<0>(network.layers)[1], 2.0, 1e-6);
};



// =============================
// ========= getResult =========
// =============================
TEST(NeuralNetworkGetResult, ReturnsLastLayerNodes){
  NN::NeuralNetwork<2, 2, 1> network;

  EXPECT_EQ(network.getResult(), std::get<1>(network.layers).getNodes());
};

TEST(NeuralNetworkGetResult, ReflectsLastLayerValues){
  NN::NeuralNetwork<2, 2, 1> network;

  std::get<1>(network.layers).setNodes({4.0, 8.0});

  float* result = network.getResult();

  ASSERT_NE(result, nullptr);
  EXPECT_NEAR(result[0], 4.0, 1e-6);
  EXPECT_NEAR(result[1], 8.0, 1e-6);
};



// =============================
// ========== Forward ==========
// =============================
TEST(NeuralNetworkForward, ForwardsThroughTwoLayers){
  NN::NeuralNetwork<2, 2, 1> network;

  network.setInput({2.0, 3.0});

  std::get<0>(network.layers).setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  network.forward();

  EXPECT_NEAR(network.getResult()[0], 11.0, 1e-6);
  EXPECT_NEAR(network.getResult()[1], 29.0, 1e-6);
};

TEST(NeuralNetworkForward, ForwardsThroughMultipleLayers){
  NN::NeuralNetwork<2, 2, 2, 1> network;

  network.setInput({1.0, 2.0});

  std::get<0>(network.layers).setWeights({
    1.0, 1.0, 1.0,
    2.0, 1.0, 0.0
  });

  std::get<1>(network.layers).setWeights({
    1.0, 2.0, 1.0,
    3.0, 1.0, 2.0
  });

  network.forward();

  EXPECT_NEAR(std::get<1>(network.layers)[0], 4.0, 1e-6);
  EXPECT_NEAR(std::get<1>(network.layers)[1], 4.0, 1e-6);

  EXPECT_NEAR(network.getResult()[0], 13.0, 1e-6);
  EXPECT_NEAR(network.getResult()[1], 18.0, 1e-6);
};

TEST(NeuralNetworkForward, AppliesActivation){
  NN::NeuralNetwork<1, 1, 1> network;

  network.setInput({0.0});
  network.setActivation<0, NN::Sigmoid>();

  std::get<0>(network.layers).setWeights({2.0, 0.0});

  network.forward();

  EXPECT_NEAR(network.getResult()[0], 1.0, 1e-6);
};

TEST(NeuralNetworkForward, IncludesBias){
  NN::NeuralNetwork<2, 1, 1> network;

  network.setInput({0.0, 0.0});
  std::get<0>(network.layers).setWeights({0.0, 0.0, 7.0});

  network.forward();

  EXPECT_NEAR(network.getResult()[0], 7.0, 1e-6);
};

TEST(NeuralNetworkForward, ProducesFiniteValues){
  NN::NeuralNetwork<3, 8, 8, 1> network;

  network.setInput({0.2, 0.4, 0.6});
  network.forward();

  float* result = network.getResult();

  ASSERT_NE(result, nullptr);

  for(unsigned int i = 0; i < 8; ++i)
    EXPECT_TRUE(std::isfinite(result[i]));
};



// =============================
// ======== forwardImpl ========
// =============================
TEST(NeuralNetworkForwardImpl, ForwardsRequestedConnections){
  NN::NeuralNetwork<2, 2, 1> network;

  network.setInput({2.0, 3.0});

  std::get<0>(network.layers).setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  network.forwardImpl(std::index_sequence<0>{});

  EXPECT_NEAR(network.getResult()[0], 11.0, 1e-6);
  EXPECT_NEAR(network.getResult()[1], 29.0, 1e-6);
};



// =============================
// ===== backpropInitial =======
// =============================
TEST(NeuralNetworkBackpropInitial, CalculatesLastLayerSigma){
  NN::NeuralNetwork<1, 1, 1> network;

  network.setInput({2.0});
  std::get<0>(network.layers).setWeights({1.0, 0.0});

  network.forward();
  network.backpropInitial({1.0});

  EXPECT_NEAR(std::get<1>(network.layers).getSigma()[0], 1.0, 1e-6);
};

TEST(NeuralNetworkBackpropInitial, CalculatesZeroSigmaForCorrectResult){
  NN::NeuralNetwork<1, 1, 1> network;

  network.setInput({2.0});
  std::get<0>(network.layers).setWeights({1.0, 0.0});

  network.forward();
  network.backpropInitial({2.0});

  EXPECT_NEAR(std::get<1>(network.layers).getSigma()[0], 0.0, 1e-6);
};



// =============================
// ======== backpropImpl =======
// =============================
TEST(NeuralNetworkBackpropImpl, CalculatesSigmaAndUpdatesWeights){
  NN::NeuralNetwork<1, 1, 1> network;

  network.setLearningRate(0.1);

  std::get<0>(network.layers).setNodes({2.0});
  std::get<0>(network.layers).setWeights({1.0, 0.0});

  std::get<1>(network.layers).sigma[0] = 1.0;

  network.backpropImpl(std::index_sequence<0>{});

  EXPECT_NEAR(std::get<0>(network.layers).getSigma()[0], 1.0, 1e-6);
  EXPECT_NEAR(std::get<0>(network.layers).getWeights()[0], 0.8, 1e-6);
  EXPECT_NEAR(std::get<0>(network.layers).getWeights()[1], -0.1, 1e-6);
};



// =============================
// ========= Backprop ==========
// =============================
TEST(NeuralNetworkBackprop, CalculatesSigmaAndUpdatesWeights){
  NN::NeuralNetwork<1, 1, 1> network;

  network.setLearningRate(0.1);
  network.setInput({2.0});

  std::get<0>(network.layers).setWeights({1.0, 0.0});

  network.forward();
  network.backprop({1.0});

  EXPECT_NEAR(std::get<1>(network.layers).getSigma()[0], 1.0, 1e-6);
  EXPECT_NEAR(std::get<0>(network.layers).getSigma()[0], 1.0, 1e-6);

  EXPECT_NEAR(std::get<0>(network.layers).getWeights()[0], 0.8, 1e-6);
  EXPECT_NEAR(std::get<0>(network.layers).getWeights()[1], -0.1, 1e-6);
};

TEST(NeuralNetworkBackprop, DoesNotChangeWeightsForZeroLoss){
  NN::NeuralNetwork<1, 1, 1> network;

  network.setLearningRate(0.1);
  network.setInput({2.0});

  std::get<0>(network.layers).setWeights({1.0, 0.0});

  network.forward();
  network.backprop({2.0});

  EXPECT_NEAR(std::get<0>(network.layers).getWeights()[0], 1.0, 1e-6);
  EXPECT_NEAR(std::get<0>(network.layers).getWeights()[1], 0.0, 1e-6);
};

TEST(NeuralNetworkBackprop, UpdatesMultipleConnections){
  NN::NeuralNetwork<1, 1, 1, 1> network;

  network.setLearningRate(0.1);
  network.setInput({1.0});

  std::get<0>(network.layers).setWeights({1.0, 0.0});
  std::get<1>(network.layers).setWeights({1.0, 0.0});

  network.forward();

  const float first_before = std::get<0>(network.layers).getWeights()[0];
  const float second_before = std::get<1>(network.layers).getWeights()[0];

  network.backprop({0.0});

  EXPECT_NEAR(std::get<0>(network.layers).getWeights()[0], 0.9, 1e-6);
  EXPECT_NEAR(std::get<1>(network.layers).getWeights()[0], 0.9, 1e-6);

  EXPECT_NE(std::get<0>(network.layers).getWeights()[0], first_before);
  EXPECT_NE(std::get<1>(network.layers).getWeights()[0], second_before);
};

TEST(NeuralNetworkBackprop, KeepsWeightsAndSigmaFinite){
  NN::NeuralNetwork<2, 4, 4, 1> network;

  network.setLearningRate(0.0001);
  network.setActivation<1, NN::Sigmoid>();

  for(unsigned int i = 0; i < 100; ++i){
    network.setInput({0.25, 0.5});
    network.forward();
    network.backprop({0.5, 0.5, 0.5, 0.5});
  };

  for(float weight : std::get<0>(network.layers).weights)
    EXPECT_TRUE(std::isfinite(weight));

  for(float weight : std::get<1>(network.layers).weights)
    EXPECT_TRUE(std::isfinite(weight));

  for(float sigma : std::get<0>(network.layers).sigma)
    EXPECT_TRUE(std::isfinite(sigma));

  for(float sigma : std::get<1>(network.layers).sigma)
    EXPECT_TRUE(std::isfinite(sigma));
};



// =============================
// ======= Integration =========
// =============================
TEST(NeuralNetworkIntegration, BackpropReducesLossForSimpleSample){
  NN::NeuralNetwork<1, 1, 1> network;

  network.setLearningRate(0.1);
  network.setInput({1.0});

  std::get<0>(network.layers).setWeights({0.0, 0.0});

  network.forward();

  const float target = 1.0;
  const float before = std::pow(network.getResult()[0] - target, 2.0) / 2.0;

  network.backprop({target});

  network.setInput({1.0});
  network.forward();

  const float after = std::pow(network.getResult()[0] - target, 2.0) / 2.0;

  EXPECT_LT(after, before);
};

TEST(NeuralNetworkIntegration, DifferentInputsProduceDifferentResults){
  NN::NeuralNetwork<2, 2, 1> network;

  std::get<0>(network.layers).setWeights({
    1.0, 2.0, 0.0,
    3.0, 4.0, 0.0
  });

  network.setInput({1.0, 2.0});
  network.forward();

  const float first_a = network.getResult()[0];
  const float first_b = network.getResult()[1];

  network.setInput({3.0, 4.0});
  network.forward();

  const float second_a = network.getResult()[0];
  const float second_b = network.getResult()[1];

  EXPECT_NE(first_a, second_a);
  EXPECT_NE(first_b, second_b);
};

TEST(NeuralNetworkIntegration, RepeatedTrainingKeepsResultFinite){
  NN::NeuralNetwork<2, 4, 4, 1> network;

  network.setLearningRate(0.0001);
  network.setActivation<1, NN::Sigmoid>();

  for(unsigned int i = 0; i < 1000; ++i){
    network.setInput({0.2, 0.4});
    network.forward();
    network.backprop({0.5, 0.5, 0.5, 0.5});
  };

  network.setInput({0.2, 0.4});
  network.forward();

  float* result = network.getResult();

  ASSERT_NE(result, nullptr);

  for(unsigned int i = 0; i < 4; ++i)
    EXPECT_TRUE(std::isfinite(result[i]));
};



// =============================
// ======== serialize ==========
// =============================
TEST(NeuralNetworkSerialize, ReturnsBinaryData){
  NN::NeuralNetwork<2, 2, 1> network;

  std::get<0>(network.layers).setWeights({
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
  });
  std::get<1>(network.layers).setWeights({
    0.7, 0.8, 0.9
  });
  network.setLearningRate(0.01);
  network.setActivation<0, NN::Sigmoid>();
  network.setLoss<1, NN::MSE>();

  std::string data = network.serialize();

  EXPECT_FALSE(data.empty());
};

TEST(NeuralNetworkSerialize, ProducesSameDataForSameNetwork){
  NN::NeuralNetwork<2, 2, 1> network1;
  NN::NeuralNetwork<2, 2, 1> network2;

  std::get<0>(network1.layers).setWeights({
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
  });
  std::get<1>(network1.layers).setWeights({
    0.7, 0.8, 0.9
  });

  std::get<0>(network2.layers).setWeights({
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
  });
  std::get<1>(network2.layers).setWeights({
    0.7, 0.8, 0.9
  });

  network1.setLearningRate(0.01);
  network2.setLearningRate(0.01);
  network1.setActivation<0, NN::Sigmoid>();
  network2.setActivation<0, NN::Sigmoid>();

  EXPECT_EQ(network1.serialize(), network2.serialize());
};



// =============================
// ======= deserialize =========
// =============================
TEST(NeuralNetworkDeserialize, RestoresWeights){
  NN::NeuralNetwork<2, 2, 1> network;

  std::get<0>(network.layers).setWeights({
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
  });
  std::get<1>(network.layers).setWeights({
    0.7, 0.8, 0.9
  });

  std::string data = network.serialize();
  NN::NeuralNetwork<2, 2, 1> network2;
  network2.deserialize(data);

  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[0], 0.1, 1e-6);
  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[1], 0.2, 1e-6);
  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[2], 0.3, 1e-6);
  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[3], 0.4, 1e-6);
  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[4], 0.5, 1e-6);
  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[5], 0.6, 1e-6);

  EXPECT_NEAR(std::get<1>(network2.layers).getWeights()[0], 0.7, 1e-6);
  EXPECT_NEAR(std::get<1>(network2.layers).getWeights()[1], 0.8, 1e-6);
  EXPECT_NEAR(std::get<1>(network2.layers).getWeights()[2], 0.9, 1e-6);
};

TEST(NeuralNetworkDeserialize, RestoresLearningRate){
  NN::NeuralNetwork<2, 2, 1> network;
  network.setLearningRate(0.123);

  std::string data = network.serialize();
  NN::NeuralNetwork<2, 2, 1> network2;
  network2.deserialize(data);

  EXPECT_NEAR(std::get<0>(network2.layers).getLearningRate(), 0.123, 1e-6);
  EXPECT_NEAR(std::get<1>(network2.layers).getLearningRate(), 0.123, 1e-6);
};

TEST(NeuralNetworkDeserialize, RestoresActivation){
  NN::NeuralNetwork<2, 2, 1> network;
  network.setActivation<0, NN::Sigmoid>();

  std::string data = network.serialize();
  NN::NeuralNetwork<2, 2, 1> network2;
  network2.deserialize(data);

  ASSERT_NE(std::get<0>(network2.layers).getActivation(), nullptr);
  EXPECT_NE(dynamic_cast<NN::Sigmoid*>(std::get<0>(network2.layers).getActivation().get()), nullptr);
  EXPECT_NE(dynamic_cast<NN::Linear*>(std::get<1>(network2.layers).getActivation().get()), nullptr);
};

TEST(NeuralNetworkDeserialize, RestoresLoss){
  NN::NeuralNetwork<2, 2, 1> network;
  network.setLoss<0, NN::MSE>();
  network.setLoss<1, NN::MSE>();

  std::string data = network.serialize();
  NN::NeuralNetwork<2, 2, 1> network2;
  network2.deserialize(data);

  ASSERT_NE(std::get<0>(network2.layers).getLoss(), nullptr);
  ASSERT_NE(std::get<1>(network2.layers).getLoss(), nullptr);
  EXPECT_NE(dynamic_cast<NN::MSE*>(std::get<0>(network2.layers).getLoss().get()), nullptr);
  EXPECT_NE(dynamic_cast<NN::MSE*>(std::get<1>(network2.layers).getLoss().get()), nullptr);
};

TEST(NeuralNetworkDeserialize, RestoresWholeNetwork){
  NN::NeuralNetwork<2, 2, 1> network;

  std::get<0>(network.layers).setWeights({
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
  });
  std::get<1>(network.layers).setWeights({
    0.7, 0.8, 0.9
  });

  network.setLearningRate(0.02);
  network.setActivation<0, NN::Sigmoid>();
  network.setLoss<0, NN::MSE>();
  network.setLoss<1, NN::MSE>();

  std::string data = network.serialize();

  NN::NeuralNetwork<2, 2, 1> network2;
  network2.deserialize(data);

  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[0], 0.1, 1e-6);
  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[1], 0.2, 1e-6);
  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[2], 0.3, 1e-6);
  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[3], 0.4, 1e-6);
  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[4], 0.5, 1e-6);
  EXPECT_NEAR(std::get<0>(network2.layers).getWeights()[5], 0.6, 1e-6);

  EXPECT_NEAR(std::get<1>(network2.layers).getWeights()[0], 0.7, 1e-6);
  EXPECT_NEAR(std::get<1>(network2.layers).getWeights()[1], 0.8, 1e-6);
  EXPECT_NEAR(std::get<1>(network2.layers).getWeights()[2], 0.9, 1e-6);

  EXPECT_NEAR(std::get<0>(network2.layers).getLearningRate(), 0.02, 1e-6);
  EXPECT_NEAR(std::get<1>(network2.layers).getLearningRate(), 0.02, 1e-6);

  EXPECT_NE(dynamic_cast<NN::Sigmoid*>(std::get<0>(network2.layers).getActivation().get()), nullptr);
  EXPECT_NE(dynamic_cast<NN::Linear*>(std::get<1>(network2.layers).getActivation().get()), nullptr);

  EXPECT_NE(dynamic_cast<NN::MSE*>(std::get<0>(network2.layers).getLoss().get()), nullptr);
  EXPECT_NE(dynamic_cast<NN::MSE*>(std::get<1>(network2.layers).getLoss().get()), nullptr);
};

TEST(NeuralNetworkDeserialize, ProducesSameSerializationAfterRoundTrip){
  NN::NeuralNetwork<2, 2, 1> network;

  std::get<0>(network.layers).setWeights({
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
  });
  std::get<1>(network.layers).setWeights({
    0.7, 0.8, 0.9
  });

  network.setLearningRate(0.02);
  network.setActivation<0, NN::Sigmoid>();

  std::string data = network.serialize();

  NN::NeuralNetwork<2, 2, 1> network2;
  network2.deserialize(data);

  EXPECT_EQ(network2.serialize(), data);
};

TEST(NeuralNetworkDeserialize, ThrowsOnDifferentLayerCount){
  NN::NeuralNetwork<2, 2, 1> network;

  std::string data = network.serialize();

  NN::NeuralNetwork<2, 2, 2, 1> network2;
  EXPECT_THROW(network2.deserialize(data), std::runtime_error);
};

TEST(NeuralNetworkDeserialize, ThrowsOnDifferentLayerSize){
  NN::NeuralNetwork<2, 2, 1> network;

  std::string data = network.serialize();

  NN::NeuralNetwork<3, 2, 1> network2;
  EXPECT_THROW(network2.deserialize(data), std::runtime_error);
};

TEST(NeuralNetworkDeserialize, ThrowsOnTruncatedData){
  NN::NeuralNetwork<2, 2, 1> network;

  std::string data = network.serialize();
  data.resize(data.size() / 2);

  NN::NeuralNetwork<2, 2, 1> network2;
  EXPECT_THROW(network2.deserialize(data), std::runtime_error);
};