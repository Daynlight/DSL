// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include "fmt/base.h"
#include "fmt/color.h"

#include <iomanip>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <functional>

#include "NeuralNetwork/Layer/Layer.h"
#include "NeuralNetwork/NeuralNetwork.h"



namespace ExamplesTemplatedCPU::Operations{
void BaseOperations(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "---- BaseOperationExamples ----");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println("");
  
  fmt::println(fg(fmt::color::yellow), "-- Create Layer");
  NN::Layer<2, 1> layer;
  fmt::println(fg(fmt::color::white), "{}", layer.print());
  
  fmt::println(fg(fmt::color::yellow), "-- Set Nodes");
  layer.setNodes({6});
  fmt::println(fg(fmt::color::white), "{}", layer.print());
  
  fmt::println(fg(fmt::color::yellow), "-- Set Weights");
  layer.setWeights({2, 4});
  fmt::println(fg(fmt::color::white), "{}", layer.print());

  fmt::println(fg(fmt::color::yellow), "-- Set Loss");
  layer.setLoss<NN::MSE>();
  fmt::println(fg(fmt::color::white), "{}", layer.print());

  fmt::println(fg(fmt::color::yellow), "-- Set Activation");
  fmt::println(fg(fmt::color::green), "---- SIGMOID");
  layer.setActivation<NN::Sigmoid>();
  fmt::println(fg(fmt::color::white), "{}", layer.print());

  fmt::println(fg(fmt::color::green), "---- LINEAR");
  layer.setActivation<NN::Linear>();
  fmt::println(fg(fmt::color::white), "{}", layer.print());
};



void Serialization(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------- Serialization --------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println("");
  
  NN::Layer<2, 1> layer;  
  layer.setNodes({6});
  layer.setWeights({2, 4});
  layer.setLoss<NN::MSE>();
  layer.setActivation<NN::Sigmoid>();

  fmt::println(fg(fmt::color::yellow), "---- Layer");
  fmt::println(fg(fmt::color::white), "{}", layer.print());

  fmt::println(fg(fmt::color::yellow), "---- Serialize");
  std::string data = layer.serialize();
  for(unsigned char byte : data) fmt::print("{:02X} ", byte);
  fmt::println("");

  fmt::println(fg(fmt::color::yellow), "---- Deserialize");
  NN::Layer<2, 1> layer2;
  layer2.deserialize(data);

  fmt::println(fg(fmt::color::yellow), "---- Layer 2");
  fmt::println(fg(fmt::color::white), "{}", layer2.print());
};



void SerializationNetwork(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "---- Serialization Network ----");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println("");
  
  std::srand(std::time(nullptr));
  NN::NeuralNetwork network = NN::NeuralNetwork<2, 2, 2, 1, 1>();

  const float learning_rate = 0.01;
  const unsigned int modulo_number = 2;
  const unsigned int learn_samples = 2000;
  const unsigned int epoch = 1000;
  const unsigned int tests = 10000;
  network.setLearningRate(learning_rate);
  network.setActivation<1, NN::Sigmoid>();
  fmt::println(fg(fmt::color::yellow), "-- Parameters:");
  fmt::println(fg(fmt::color::white), "learning_rate = {}", learning_rate);
  fmt::println(fg(fmt::color::white), "modulo_number = {}", modulo_number);
  fmt::println(fg(fmt::color::white), "learn_samples = {}", learn_samples);
  fmt::println(fg(fmt::color::white), "tests = {}", tests);
  fmt::println(fg(fmt::color::white), "epoch = {}", epoch);

  fmt::println(fg(fmt::color::yellow), "-- Learning from random set");
  for(unsigned int j = 0; j < epoch; j++){
    for(unsigned int i = 0; i < learn_samples; i++) {
      float x = (rand()%modulo_number);
      float y = (rand()%modulo_number);

      NN::Utils::progressBar(i + j * learn_samples, learn_samples * epoch);

      network.setInput({x, y});
      network.forward();
      float res = ((x != 0) != (y != 0));
      network.backprop({res});
    };
  };

  fmt::println(fg(fmt::color::yellow), "-- Testing on random set");
  float sum = 0;
  for(unsigned int i = 0; i < tests; i++) {
    float x = (rand()%modulo_number);
    float y = (rand()%modulo_number);

    NN::Utils::progressBar(i, tests);

    network.setInput({x, y});
    network.forward();
    float res = ((x != 0) != (y != 0));
    if ((network.getResult()[0] >= 0.5) == res) sum += 1;
  };
  if((sum / tests) * 100 >= 90) fmt::println(fg(fmt::color::green), "avg: {}%", (sum / tests) * 100);
  else                          fmt::println(fg(fmt::color::red), "avg: {}%", (sum / tests) * 100);


  fmt::println(fg(fmt::color::yellow), "---- Network");
  fmt::println(fg(fmt::color::white), "{}", network.print());

  fmt::println(fg(fmt::color::yellow), "---- Serialize");
  std::string data = network.serialize();
  for(unsigned char byte : data) fmt::print("{:02X} ", byte);
  fmt::println("");

  fmt::println(fg(fmt::color::yellow), "---- Deserialize");
  NN::NeuralNetwork network2 = NN::NeuralNetwork<2, 2, 2, 1, 1>();
  network2.deserialize(data);

  fmt::println(fg(fmt::color::yellow), "---- Network 2");
  fmt::println(fg(fmt::color::white), "{}", network2.print());

  fmt::println(fg(fmt::color::yellow), "-- Testing Network 2 on random set");
  sum = 0;
  for(unsigned int i = 0; i < tests; i++) {
    float x = (rand()%modulo_number);
    float y = (rand()%modulo_number);

    NN::Utils::progressBar(i, tests);

    network2.setInput({x, y});
    network2.forward();
    float res = ((x != 0) != (y != 0));
    if ((network2.getResult()[0] >= 0.5) == res) sum += 1;
  };
  if((sum / tests) * 100 >= 90) fmt::println(fg(fmt::color::green), "avg: {}%", (sum / tests) * 100);
  else                          fmt::println(fg(fmt::color::red), "avg: {}%", (sum / tests) * 100);
};



void Forward(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "----------- Forward -----------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println("");

  NN::Layer<4, 3> a;
  fmt::println(fg(fmt::color::purple), "a => {}", a.print());
  NN::Layer<3, 0> b;
  fmt::println(fg(fmt::color::pink), "b => {}", b.print());

  a.setNodes({5, 2, 1, 3});
  a.setWeights({1, 2 ,1, -1, 5,   2, 1, 3, 1, 0,   1, -4, 2, 5, 6});
  fmt::println(fg(fmt::color::purple), "a initialize => {}", a.print());
  
  a.forward(b);
  fmt::println(fg(fmt::color::yellow), "a => forward => b");
  fmt::println(fg(fmt::color::white), "5 * 1 + 2 * 2 + 1 * 1 + 3 * (-1) + 5 = 12");
  fmt::println(fg(fmt::color::white), "5 * 2 + 2 * 1 + 1 * 3 + 3 * 1 + 0 = 18");
  fmt::println(fg(fmt::color::white), "5 * 1 + 2 * (-4) + 1 * 2 + 3 * 5 + 6 = 20"); 

  fmt::println(fg(fmt::color::purple), "a => {}", a.print());
  fmt::println(fg(fmt::color::pink), "b => {}", b.print());
};



void Backprop(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "----------- Backprop ----------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "-------------------------------");
  fmt::println("");

  NN::Layer<2, 2> c;
  NN::Layer<2, 2> h;
  NN::Layer<2, 0> d;
  c.setWeights({1,1,1,  1,1,1});
  h.setWeights({1,1,1,   1,1,1});
  c.setNodes({2, 1});

  fmt::println(fg(fmt::color::pink), "c => {}", c.print());
  fmt::println(fg(fmt::color::purple), "h => {}", h.print());
  fmt::println(fg(fmt::color::violet), "d => {}", d.print());
  
  
  c.forward(h);
  h.forward(d);
  fmt::println(fg(fmt::color::yellow), "c => forward => h => forward => d");

  fmt::println(fg(fmt::color::pink), "c => {}", c.print());
  fmt::println(fg(fmt::color::purple), "h => {}", h.print());
  fmt::println(fg(fmt::color::violet), "d => {}", d.print());

  d.backprop_initial({3, 3, 1});
  h.backprop(d);
  c.backprop(h);
  fmt::println(fg(fmt::color::yellow), "c <= backprop <= h <= backprop <= d");

  fmt::println(fg(fmt::color::pink), "c => {}", c.print());
  fmt::println(fg(fmt::color::purple), "h => {}", h.print());
  fmt::println(fg(fmt::color::violet), "d => {}", d.print());
};
};