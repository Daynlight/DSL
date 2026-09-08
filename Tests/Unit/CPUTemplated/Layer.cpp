// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>
#include <array>
#include <algorithm>
#include <cmath>
#include <type_traits>

#include "NeuralNetwork/Layer/Layer.h"



// =============================
// ======= Constructors ========
// =============================
TEST(LayerConstructor, InitializesWeightsInExpectedRange){
  NN::Layer<2, 2> layer;

  float* weights = layer.getWeights();

  for(unsigned int i = 0; i < 6; i++){
    EXPECT_GE(weights[i], -1.0);
    EXPECT_LE(weights[i], 1.0);
  };
};

TEST(LayerConstructor, InitializesDefaultLearningRate){
  NN::Layer<2, 2> layer;

  EXPECT_NEAR(layer.getLearningRate(), 0.005, 1e-6);
};

TEST(LayerConstructor, InitializesDefaultLinearActivation){
  NN::Layer<2, 2> layer;

  ASSERT_NE(layer.getActivation(), nullptr);
  EXPECT_NE(dynamic_cast<NN::Linear*>(layer.getActivation().get()), nullptr);
};

TEST(LayerConstructor, InitializesDefaultMSELoss){
  NN::Layer<2, 2> layer;

  ASSERT_NE(layer.getLoss(), nullptr);
  EXPECT_NE(dynamic_cast<NN::MSE*>(layer.getLoss().get()), nullptr);
};

TEST(LayerDestructor, DestroysLayer){
  {
    NN::Layer<2, 2> layer;
  };

  SUCCEED();
};



// =============================
// ========= getNodes ==========
// =============================
TEST(LayerGetNodes, ReturnsInternalNodes){
  NN::Layer<2, 1> layer;

  layer.setNodes({1.0, 2.0});

  float* nodes = layer.getNodes();

  EXPECT_NEAR(nodes[0], 1.0, 1e-6);
  EXPECT_NEAR(nodes[1], 2.0, 1e-6);
};



// =============================
// ========= setNodes ==========
// =============================
TEST(LayerSetNodes, SetsNodes){
  NN::Layer<3, 1> layer;

  layer.setNodes({1.0, 2.0, 3.0});

  EXPECT_NEAR(layer.getNodes()[0], 1.0, 1e-6);
  EXPECT_NEAR(layer.getNodes()[1], 2.0, 1e-6);
  EXPECT_NEAR(layer.getNodes()[2], 3.0, 1e-6);
};

TEST(LayerSetNodes, StopsAtLayerSize){
  NN::Layer<2, 1> layer;

  layer.setNodes({1.0, 2.0, 3.0, 4.0});

  EXPECT_NEAR(layer.getNodes()[0], 1.0, 1e-6);
  EXPECT_NEAR(layer.getNodes()[1], 2.0, 1e-6);
};



// =============================
// ==== getActivatedNode =======
// =============================
TEST(LayerGetActivatedNode, ReturnsLinearActivatedNode){
  NN::Layer<2, 1> layer;

  layer.setNodes({2.5, -3.5});

  EXPECT_NEAR(layer.getActivatedNode(0), 2.5, 1e-6);
  EXPECT_NEAR(layer.getActivatedNode(1), -3.5, 1e-6);
};

TEST(LayerGetActivatedNode, ReturnsSigmoidActivatedNode){
  NN::Layer<2, 1> layer;

  layer.setActivation<NN::Sigmoid>();
  layer.setNodes({0.0, 1.0});

  EXPECT_NEAR(layer.getActivatedNode(0), 0.5, 1e-6);
  EXPECT_NEAR(layer.getActivatedNode(1), 0.7310585786300049, 1e-6);
};



// =============================
// ========= getWeights ========
// =============================
TEST(LayerGetWeights, ReturnsInternalWeights){
  NN::Layer<2, 2> layer;

  layer.setWeights({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  float* weights = layer.getWeights();

  EXPECT_NEAR(weights[0], 1.0, 1e-6);
  EXPECT_NEAR(weights[1], 2.0, 1e-6);
  EXPECT_NEAR(weights[2], 3.0, 1e-6);
  EXPECT_NEAR(weights[3], 4.0, 1e-6);
  EXPECT_NEAR(weights[4], 5.0, 1e-6);
  EXPECT_NEAR(weights[5], 6.0, 1e-6);
};



// =============================
// ======== setWeights =========
// =============================
TEST(LayerSetWeights, SetsWeightsFromInitializerList){
  NN::Layer<2, 2> layer;

  layer.setWeights({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  EXPECT_NEAR(layer.getWeights()[0], 1.0, 1e-6);
  EXPECT_NEAR(layer.getWeights()[1], 2.0, 1e-6);
  EXPECT_NEAR(layer.getWeights()[2], 3.0, 1e-6);
  EXPECT_NEAR(layer.getWeights()[3], 4.0, 1e-6);
  EXPECT_NEAR(layer.getWeights()[4], 5.0, 1e-6);
  EXPECT_NEAR(layer.getWeights()[5], 6.0, 1e-6);
};

TEST(LayerSetWeights, StopsAtWeightCount){
  NN::Layer<1, 1> layer;

  layer.setWeights({1.0, 2.0, 3.0, 4.0});

  EXPECT_NEAR(layer.getWeights()[0], 1.0, 1e-6);
  EXPECT_NEAR(layer.getWeights()[1], 2.0, 1e-6);
};

TEST(LayerSetWeights, SetsWeightsFromPointer){
  NN::Layer<2, 2> layer;

  const float weights[] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};

  layer.setWeights(weights);

  EXPECT_NEAR(layer.getWeights()[0], 6.0, 1e-6);
  EXPECT_NEAR(layer.getWeights()[1], 5.0, 1e-6);
  EXPECT_NEAR(layer.getWeights()[2], 4.0, 1e-6);
  EXPECT_NEAR(layer.getWeights()[3], 3.0, 1e-6);
  EXPECT_NEAR(layer.getWeights()[4], 2.0, 1e-6);
  EXPECT_NEAR(layer.getWeights()[5], 1.0, 1e-6);
};



// =============================
// ==== getLearningRate ========
// =============================
TEST(LayerGetLearningRate, ReturnsLearningRate){
  NN::Layer<2, 1> layer;

  layer.setLearningRate(0.123);

  EXPECT_NEAR(layer.getLearningRate(), 0.123, 1e-6);
};



// =============================
// ==== setLearningRate ========
// =============================
TEST(LayerSetLearningRate, SetsLearningRate){
  NN::Layer<2, 1> layer;

  layer.setLearningRate(0.01);

  EXPECT_NEAR(layer.getLearningRate(), 0.01, 1e-6);
};

TEST(LayerSetLearningRate, AcceptsZero){
  NN::Layer<2, 1> layer;

  layer.setLearningRate(0.0);

  EXPECT_NEAR(layer.getLearningRate(), 0.0, 1e-6);
};



// =============================
// ====== getActivation ========
// =============================
TEST(LayerGetActivation, ReturnsActivation){
  NN::Layer<2, 1> layer;

  const std::unique_ptr<NN::iActivation>& activation = layer.getActivation();

  ASSERT_NE(activation, nullptr);
  EXPECT_NE(dynamic_cast<NN::Linear*>(activation.get()), nullptr);
};



// =============================
// ====== setActivation ========
// =============================
TEST(LayerSetActivation, SetsSigmoid){
  NN::Layer<2, 1> layer;

  layer.setActivation<NN::Sigmoid>();

  ASSERT_NE(layer.getActivation(), nullptr);
  EXPECT_NE(dynamic_cast<NN::Sigmoid*>(layer.getActivation().get()), nullptr);
};

TEST(LayerSetActivation, ReplacesPreviousActivation){
  NN::Layer<2, 1> layer;

  layer.setActivation<NN::Sigmoid>();
  EXPECT_NE(dynamic_cast<NN::Sigmoid*>(layer.getActivation().get()), nullptr);

  layer.setActivation<NN::Linear>();
  EXPECT_NE(dynamic_cast<NN::Linear*>(layer.getActivation().get()), nullptr);
};



// =============================
// ========= getLoss ===========
// =============================
TEST(LayerGetLoss, ReturnsLoss){
  NN::Layer<2, 1> layer;

  const std::unique_ptr<NN::iLoss>& loss = layer.getLoss();

  ASSERT_NE(loss, nullptr);
  EXPECT_NE(dynamic_cast<NN::MSE*>(loss.get()), nullptr);
};



// =============================
// ========= setLoss ===========
// =============================
TEST(LayerSetLoss, SetsMSE){
  NN::Layer<2, 1> layer;

  layer.setLoss<NN::MSE>();

  ASSERT_NE(layer.getLoss(), nullptr);
  EXPECT_NE(dynamic_cast<NN::MSE*>(layer.getLoss().get()), nullptr);
};



// =============================
// ======== operator[] =========
// =============================
TEST(LayerOperatorIndex, ReturnsNodeReference){
  NN::Layer<2, 1> layer;

  layer.setNodes({1.0, 2.0});

  EXPECT_NEAR(layer[0], 1.0, 1e-6);
  EXPECT_NEAR(layer[1], 2.0, 1e-6);
};

TEST(LayerOperatorIndex, AllowsModifyingNode){
  NN::Layer<2, 1> layer;

  layer[0] = 42.0;

  EXPECT_NEAR(layer.getNodes()[0], 42.0, 1e-6);
};

TEST(LayerOperatorIndex, AllowsBiasIndex){
  NN::Layer<2, 1> layer;

  layer[2] = 1.0;

  EXPECT_NEAR(layer[2], 1.0, 1e-6);
};

TEST(LayerOperatorIndex, ThrowsForOutOfRangeIndex){
  NN::Layer<2, 1> layer;

  EXPECT_THROW(layer[3], std::range_error);
  EXPECT_THROW(layer[100], std::range_error);
};



// =============================
// ========= getSigma ==========
// =============================
TEST(LayerGetSigma, ReturnsCalculatedSigma){
  NN::Layer<2, 1> layer;

  layer.setNodes({3.0, 5.0});
  layer.backprop_initial({1.0, 2.0});

  const float* sigma = layer.getSigma();

  EXPECT_NEAR(sigma[0], 2.0, 1e-6);
  EXPECT_NEAR(sigma[1], 3.0, 1e-6);
};



// =============================
// ========= forward ===========
// =============================
TEST(LayerForward, CalculatesExpectedValues){
  NN::Layer<2, 2> layer;
  NN::Layer<2, 1> next;

  layer.setNodes({1.0, 2.0});
  layer.setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  layer.forward(next);

  EXPECT_NEAR(next.getNodes()[0], 8.0, 1e-6);
  EXPECT_NEAR(next.getNodes()[1], 20.0, 1e-6);
};

TEST(LayerForward, SetsCurrentBiasToOne){
  NN::Layer<2, 1> layer;
  NN::Layer<1, 1> next;

  layer.setNodes({1.0, 2.0});
  layer.setWeights({1.0, 1.0, 1.0});

  layer.forward(next);

  EXPECT_NEAR(layer[2], 1.0, 1e-6);
};

TEST(LayerForward, SetsNextLayerBiasToOne){
  NN::Layer<2, 2> layer;
  NN::Layer<2, 1> next;

  layer.setNodes({1.0, 2.0});
  layer.setWeights({
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0
  });

  layer.forward(next);

  EXPECT_NEAR(next[2], 1.0, 1e-6);
};

TEST(LayerForward, DoesNotActivateBias){
  NN::Layer<2, 1> layer;
  NN::Layer<1, 1> next;

  layer.setActivation<NN::Sigmoid>();
  layer.setNodes({0.0, 0.0});
  layer.setWeights({0.0, 0.0, 2.0});

  layer.forward(next);

  EXPECT_NEAR(next.getNodes()[0], 2.0, 1e-6);
};

TEST(LayerForward, AppliesActivationToNodes){
  NN::Layer<2, 1> layer;
  NN::Layer<1, 1> next;

  layer.setActivation<NN::Sigmoid>();
  layer.setNodes({0.0, 0.0});
  layer.setWeights({1.0, 1.0, 0.0});

  layer.forward(next);

  EXPECT_NEAR(next.getNodes()[0], 1.0, 1e-6);
};



// =============================
// ==== backprop_initial =======
// =============================
TEST(LayerBackpropInitial, CalculatesLinearSigma){
  NN::Layer<2, 1> layer;

  layer.setNodes({3.0, 5.0});

  layer.backprop_initial({1.0, 2.0});

  EXPECT_NEAR(layer.getSigma()[0], 2.0, 1e-6);
  EXPECT_NEAR(layer.getSigma()[1], 3.0, 1e-6);
};

TEST(LayerBackpropInitial, CalculatesSigmoidSigma){
  NN::Layer<2, 1> layer;

  layer.setActivation<NN::Sigmoid>();
  layer.setNodes({0.0, 0.0});

  layer.backprop_initial({0.0, 1.0});

  EXPECT_NEAR(layer.getSigma()[0], 0.125, 1e-6);
  EXPECT_NEAR(layer.getSigma()[1], -0.125, 1e-6);
};

TEST(LayerBackpropInitial, StopsAtTargetSize){
  NN::Layer<2, 1> layer;

  layer.setNodes({3.0, 5.0});

  layer.backprop_initial({1.0});

  EXPECT_NEAR(layer.getSigma()[0], 2.0, 1e-6);
};



// =============================
// ========= backprop ==========
// =============================
TEST(LayerBackprop, CalculatesExpectedSigma){
  NN::Layer<2, 2> layer;
  NN::Layer<2, 1> next;

  layer.setNodes({1.0, 2.0});
  layer.setWeights({
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
  });

  next.setNodes({3.0, 4.0});
  next.backprop_initial({1.0, 2.0});

  layer.backprop(next);

  EXPECT_NEAR(layer.getSigma()[0], 1.0, 1e-6);
  EXPECT_NEAR(layer.getSigma()[1], 1.4, 1e-6);
};

TEST(LayerBackprop, UpdatesExpectedWeights){
  NN::Layer<2, 2> layer;
  NN::Layer<2, 1> next;

  layer.setLearningRate(0.01);
  layer.setNodes({1.0, 2.0});
  layer.setWeights({
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
  });

  next.setNodes({3.0, 4.0});
  next.backprop_initial({1.0, 2.0});

  layer.backprop(next);

  EXPECT_NEAR(layer.getWeights()[0], 0.08, 1e-6);
  EXPECT_NEAR(layer.getWeights()[1], 0.16, 1e-6);
  EXPECT_NEAR(layer.getWeights()[2], 0.28, 1e-6);

  EXPECT_NEAR(layer.getWeights()[3], 0.38, 1e-6);
  EXPECT_NEAR(layer.getWeights()[4], 0.46, 1e-6);
  EXPECT_NEAR(layer.getWeights()[5], 0.58, 1e-6);
};

TEST(LayerBackprop, UpdatesBiasWithoutActivation){
  NN::Layer<1, 1> layer;
  NN::Layer<1, 1> next;

  layer.setLearningRate(0.1);
  layer.setNodes({2.0});
  layer.setWeights({1.0, 1.0});

  next.setNodes({2.0});
  next.backprop_initial({1.0});

  layer.backprop(next);

  EXPECT_NEAR(layer.getWeights()[0], 0.8, 1e-6);
  EXPECT_NEAR(layer.getWeights()[1], 0.9, 1e-6);
};

TEST(LayerBackprop, ZeroLearningRateDoesNotModifyWeights){
  NN::Layer<2, 1> layer;
  NN::Layer<1, 1> next;

  layer.setLearningRate(0.0);
  layer.setNodes({1.0, 2.0});
  layer.setWeights({0.1, 0.2, 0.3});

  next.setNodes({3.0});
  next.backprop_initial({1.0});

  layer.backprop(next);

  EXPECT_NEAR(layer.getWeights()[0], 0.1, 1e-6);
  EXPECT_NEAR(layer.getWeights()[1], 0.2, 1e-6);
  EXPECT_NEAR(layer.getWeights()[2], 0.3, 1e-6);
};



// =============================
// ========== print ============
// =============================
TEST(LayerPrint, ReturnsExpectedRepresentation){
  NN::Layer<2, 2> layer;

  layer.setNodes({1.0, 2.0});
  layer[2] = 1.0;

  layer.setWeights({
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
  });

  const std::string expected =
    "size: \n"
    "2, 2; "
    "\nnodes: \n"
    "[1.000000, 2.000000, 1.000000]; "
    "\nweights: \n"
    "[[1.000000, 2.000000, 3.000000], \n"
    "[4.000000, 5.000000, 6.000000]];\n ";

  EXPECT_EQ(layer.print(), expected);
};



// =============================
// ======== serialize ==========
// =============================
TEST(LayerSerialize, ReturnsBinaryData){
  NN::Layer<2, 1> layer;

  layer.setWeights({0.1, 0.2, 0.3});
  layer.setLearningRate(0.01);
  layer.setActivation<NN::Sigmoid>();
  layer.setLoss<NN::MSE>();
  std::string data = layer.serialize();

  EXPECT_FALSE(data.empty());
};

TEST(LayerSerialize, ProducesSameDataForSameLayer){
  NN::Layer<2, 1> layer1;
  NN::Layer<2, 1> layer2;

  layer1.setWeights({0.1, 0.2, 0.3});
  layer2.setWeights({0.1, 0.2, 0.3});
  layer1.setLearningRate(0.01);
  layer2.setLearningRate(0.01);
  layer1.setActivation<NN::Sigmoid>();
  layer2.setActivation<NN::Sigmoid>();
  layer1.setLoss<NN::MSE>();
  layer2.setLoss<NN::MSE>();

  EXPECT_EQ(layer1.serialize(), layer2.serialize());
};



// =============================
// ======= deserialize =========
// =============================
TEST(LayerDeserialize, RestoresWeights){
  NN::Layer<2, 1> layer;
  layer.setWeights({0.1, 0.2, 0.3});

  std::string data = layer.serialize();
  NN::Layer<2, 1> layer2;
  layer2.deserialize(data);

  EXPECT_NEAR(layer2.getWeights()[0], 0.1, 1e-6);
  EXPECT_NEAR(layer2.getWeights()[1], 0.2, 1e-6);
  EXPECT_NEAR(layer2.getWeights()[2], 0.3, 1e-6);
};

TEST(LayerDeserialize, RestoresLearningRate){
  NN::Layer<2, 1> layer;
  layer.setLearningRate(0.123);

  std::string data = layer.serialize();
  NN::Layer<2, 1> layer2;
  layer2.deserialize(data);

  EXPECT_NEAR(layer2.getLearningRate(), 0.123, 1e-6);
};

TEST(LayerDeserialize, RestoresActivation){
  NN::Layer<2, 1> layer;

  layer.setActivation<NN::Sigmoid>();
  std::string data = layer.serialize();
  NN::Layer<2, 1> layer2;
  layer2.deserialize(data);

  ASSERT_NE(layer2.getActivation(), nullptr);
  EXPECT_NE(dynamic_cast<NN::Sigmoid*>(layer2.getActivation().get()), nullptr);
};

TEST(LayerDeserialize, RestoresLoss){
  NN::Layer<2, 1> layer;

  layer.setLoss<NN::MSE>();
  std::string data = layer.serialize();
  NN::Layer<2, 1> layer2;

  layer2.deserialize(data);

  ASSERT_NE(layer2.getLoss(), nullptr);
  EXPECT_NE(dynamic_cast<NN::MSE*>(layer2.getLoss().get()), nullptr);

};

TEST(LayerDeserialize, RestoresWholeLayer){
  NN::Layer<2, 2> layer;

  layer.setWeights({
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
  });

  layer.setLearningRate(0.02);
  layer.setActivation<NN::Sigmoid>();
  layer.setLoss<NN::MSE>();

  std::string data = layer.serialize();

  NN::Layer<2, 2> layer2;
  layer2.deserialize(data);

  EXPECT_NEAR(layer2.getWeights()[0], 0.1, 1e-6);
  EXPECT_NEAR(layer2.getWeights()[1], 0.2, 1e-6);
  EXPECT_NEAR(layer2.getWeights()[2], 0.3, 1e-6);
  EXPECT_NEAR(layer2.getWeights()[3], 0.4, 1e-6);
  EXPECT_NEAR(layer2.getWeights()[4], 0.5, 1e-6);
  EXPECT_NEAR(layer2.getWeights()[5], 0.6, 1e-6);
  EXPECT_NEAR(layer2.getLearningRate(), 0.02, 1e-6);
  EXPECT_NE(dynamic_cast<NN::Sigmoid*>(layer2.getActivation().get()), nullptr);
  EXPECT_NE(dynamic_cast<NN::MSE*>(layer2.getLoss().get()), nullptr);
};

TEST(LayerDeserialize, ThrowsOnDifferentSourceSize){
  NN::Layer<2, 1> layer;

  layer.setWeights({0.1, 0.2, 0.3});
  std::string data = layer.serialize();

  NN::Layer<3, 1> layer2;
  EXPECT_THROW(layer2.deserialize(data), std::runtime_error);
};

TEST(LayerDeserialize, ThrowsOnTruncatedData){
  NN::Layer<2, 1> layer;
  
  layer.setWeights({0.1, 0.2, 0.3});
  std::string data = layer.serialize();
  data.resize(data.size() / 2);

  NN::Layer<2, 1> layer2;
  EXPECT_THROW(layer2.deserialize(data), std::runtime_error);
};



// =============================
// ======= Mathematics =========
// =============================

template<typename Activation, typename OutputActivation, typename Loss>
void checkLayerMath(bool gpu){
  NN::Layer<3, 3> layer;
  NN::Layer<3, 1> next;

  layer.setGpuAcceleration(gpu);
  next.setGpuAcceleration(gpu);

  layer.setActivation<Activation>();
  next.setActivation<OutputActivation>();
  next.setLoss<Loss>();

  layer.setLearningRate(0.0f);

  std::array<float, 3> input{0.2f, -0.4f, 0.7f};
  std::array<float, 3> target{0.0f, 1.0f, 0.0f};
  std::array<float, 12> weights{
    0.2f, -0.1f, 0.3f, 0.1f,
    -0.3f, 0.2f, 0.1f, -0.2f,
    0.1f, 0.4f, -0.2f, 0.05f
  };

  layer.setNodes(input);
  layer.setWeights(weights.data());

  auto error = [&](){
    const float* output = next.getActivatedNodes();
    double value = 0.0;

    for(size_t i = 0; i < 3; i++){
      if constexpr(std::is_same_v<Loss, NN::CrossEntropy>){
        value -= double(target[i]) * std::log(std::max(double(output[i]), 1e-7));
      }
      else{
        double difference = double(output[i]) - target[i];
        value += difference * difference / 2.0;
      };
    };

    return value;
  };

  auto evaluate = [&](){
    layer.forward(next);
    return error();
  };

  layer.forward(next);
  next.backprop_initial(target);

  std::array<float, 3> output_sigma{};
  std::copy_n(next.getSigma(), 3, output_sigma.begin());

  layer.backprop(next);

  std::array<float, 3> input_sigma{};
  std::array<float, 3> activated{};

  std::copy_n(layer.getSigma(), 3, input_sigma.begin());
  std::copy_n(layer.getActivatedNodes(), 3, activated.begin());

  constexpr float epsilon = 0.001f;
  constexpr double tolerance = 0.003;

  SCOPED_TRACE(gpu ? "GPU" : "CPU");

  std::array<float, 3> output_nodes{};
  std::copy_n(next.getNodes(), 3, output_nodes.begin());

  for(size_t i = 0; i < 3; i++){
    auto plus = output_nodes;
    plus[i] += epsilon;
    next.setNodes(plus);
    double positive = error();

    auto minus = output_nodes;
    minus[i] -= epsilon;
    next.setNodes(minus);
    double negative = error();

    double numerical = (positive - negative) / (2.0 * epsilon);

    SCOPED_TRACE(i);
    ASSERT_TRUE(std::isfinite(numerical));
    ASSERT_TRUE(std::isfinite(output_sigma[i]));
    EXPECT_NEAR(output_sigma[i], numerical, tolerance);
  };

  next.setNodes(output_nodes);

  for(size_t i = 0; i < 3; i++){
    auto plus = input;
    plus[i] += epsilon;
    layer.setNodes(plus);
    double positive = evaluate();

    auto minus = input;
    minus[i] -= epsilon;
    layer.setNodes(minus);
    double negative = evaluate();

    double numerical = (positive - negative) / (2.0 * epsilon);

    SCOPED_TRACE(i);
    ASSERT_TRUE(std::isfinite(numerical));
    ASSERT_TRUE(std::isfinite(input_sigma[i]));
    EXPECT_NEAR(input_sigma[i], numerical, tolerance);
  };

  layer.setNodes(input);

  for(size_t i = 0; i < weights.size(); i++){
    auto plus = weights;
    plus[i] += epsilon;
    layer.setWeights(plus.data());
    double positive = evaluate();

    auto minus = weights;
    minus[i] -= epsilon;
    layer.setWeights(minus.data());
    double negative = evaluate();

    double numerical = (positive - negative) / (2.0 * epsilon);

    const size_t row = i / 4;
    const size_t col = i % 4;
    double analytical = output_sigma[row] * (col == 3 ? 1.0f : activated[col]);

    SCOPED_TRACE(i);
    ASSERT_TRUE(std::isfinite(numerical));
    ASSERT_TRUE(std::isfinite(analytical));
    EXPECT_NEAR(analytical, numerical, tolerance);
  };

  layer.setNodes(input);
  layer.setWeights(weights.data());
  layer.setLearningRate(0.01f);

  layer.forward(next);
  next.backprop_initial(target);
  layer.backprop(next);

  const float* updated = layer.getWeights();

  for(size_t i = 0; i < weights.size(); i++){
    const size_t row = i / 4;
    const size_t col = i % 4;
    double gradient = output_sigma[row] * (col == 3 ? 1.0f : activated[col]);
    double expected = weights[i] - 0.01 * gradient;

    SCOPED_TRACE(i);
    ASSERT_TRUE(std::isfinite(updated[i]));
    EXPECT_NEAR(updated[i], expected, 1e-5);
  };
};



TEST(LayerMathematics, NumericalGradients){
  for(bool gpu : {false, true}){
    checkLayerMath<NN::Linear, NN::Linear, NN::MSE>(gpu);
    checkLayerMath<NN::Sigmoid, NN::Sigmoid, NN::MSE>(gpu);
    checkLayerMath<NN::ReLU, NN::Linear, NN::MSE>(gpu);
    checkLayerMath<NN::Softmax, NN::Linear, NN::MSE>(gpu);
    checkLayerMath<NN::Linear, NN::ReLU, NN::MSE>(gpu);
    checkLayerMath<NN::Linear, NN::Softmax, NN::CrossEntropy>(gpu);
    checkLayerMath<NN::Sigmoid, NN::Softmax, NN::CrossEntropy>(gpu);
  };
};



// =============================
// ======= CPU/GPU Parity ======
// =============================

template<unsigned int S, unsigned int D>
void checkLayerGPUParity(){
  NN::Layer<S, D> cpu;
  NN::Layer<S, D> gpu;
  NN::Layer<D, 1> cpu_next;
  NN::Layer<D, 1> gpu_next;

  cpu.setGpuAcceleration(false);
  cpu_next.setGpuAcceleration(false);
  gpu.setGpuAcceleration(true);
  gpu_next.setGpuAcceleration(true);

  cpu.template setActivation<NN::ReLU>();
  gpu.template setActivation<NN::ReLU>();
  cpu_next.template setActivation<NN::Softmax>();
  gpu_next.template setActivation<NN::Softmax>();
  cpu_next.template setLoss<NN::CrossEntropy>();
  gpu_next.template setLoss<NN::CrossEntropy>();

  cpu.setLearningRate(0.01f);
  gpu.setLearningRate(0.01f);

  std::array<float, S> input{};
  std::array<float, (S + 1) * D> weights{};
  std::array<float, D> target{};

  for(size_t i = 0; i < S; i++)
    input[i] = 0.1f * (int(i % 7) - 3);

  for(size_t j = 0; j < D; j++){
    for(size_t i = 0; i < S; i++)
      weights[j * (S + 1) + i] = 0.015f * (int((i * 3 + j * 5) % 11) - 5);

    weights[j * (S + 1) + S] = 0.02f * (int(j % 5) - 2);
  };

  target[D / 2] = 1.0f;

  cpu.setNodes(input);
  gpu.setNodes(input);
  cpu.setWeights(weights.data());
  gpu.setWeights(weights.data());

  cpu.forward(cpu_next);
  gpu.forward(gpu_next);

  const float* cpu_output = cpu_next.getActivatedNodes();
  const float* gpu_output = gpu_next.getActivatedNodes();

  for(size_t i = 0; i < D; i++){
    SCOPED_TRACE(i);
    ASSERT_TRUE(std::isfinite(cpu_output[i]));
    ASSERT_TRUE(std::isfinite(gpu_output[i]));
    EXPECT_NEAR(cpu_output[i], gpu_output[i], 1e-4);
  };

  cpu_next.backprop_initial(target);
  gpu_next.backprop_initial(target);

  const float* cpu_output_sigma = cpu_next.getSigma();
  const float* gpu_output_sigma = gpu_next.getSigma();

  for(size_t i = 0; i < D; i++){
    SCOPED_TRACE(i);
    EXPECT_NEAR(cpu_output_sigma[i], gpu_output_sigma[i], 1e-4);
  };

  cpu.backprop(cpu_next);
  gpu.backprop(gpu_next);

  const float* cpu_sigma = cpu.getSigma();
  const float* gpu_sigma = gpu.getSigma();

  for(size_t i = 0; i < S; i++){
    SCOPED_TRACE(i);
    ASSERT_TRUE(std::isfinite(cpu_sigma[i]));
    ASSERT_TRUE(std::isfinite(gpu_sigma[i]));
    EXPECT_NEAR(cpu_sigma[i], gpu_sigma[i], 1e-4);
  };

  const float* cpu_weights = cpu.getWeights();
  const float* gpu_weights = gpu.getWeights();

  for(size_t i = 0; i < weights.size(); i++){
    SCOPED_TRACE(i);
    ASSERT_TRUE(std::isfinite(cpu_weights[i]));
    ASSERT_TRUE(std::isfinite(gpu_weights[i]));
    EXPECT_NEAR(cpu_weights[i], gpu_weights[i], 1e-4);
  };
};

// TEST(LayerMathematics, CPUAndGPUParity){
//   checkLayerGPUParity<33, 5>();
//   checkLayerGPUParity<257, 3>();
//   checkLayerGPUParity<3, 257>();
// };
