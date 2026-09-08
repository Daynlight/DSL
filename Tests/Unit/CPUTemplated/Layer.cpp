// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>

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

  EXPECT_NEAR(next[0], 8.0, 1e-6);
  EXPECT_NEAR(next[1], 20.0, 1e-6);
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

  EXPECT_NEAR(next[0], 2.0, 1e-6);
};

TEST(LayerForward, AppliesActivationToNodes){
  NN::Layer<2, 1> layer;
  NN::Layer<1, 1> next;

  layer.setActivation<NN::Sigmoid>();
  layer.setNodes({0.0, 0.0});
  layer.setWeights({1.0, 1.0, 0.0});

  layer.forward(next);

  EXPECT_NEAR(next[0], 1.0, 1e-6);
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