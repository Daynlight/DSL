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
#include "NeuralNetwork/Utils.h"



namespace ExamplesTemplatedCPU::App{
void Minus(){
  fmt::println("");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "----------------------------------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "---------- LearnMinusApp ---------");
  fmt::println(fg(fmt::color::aquamarine) | fmt::emphasis::bold, "----------------------------------");
  fmt::println("");

  std::srand(std::time(nullptr));
  NN::Layer<2, 1> e;
  NN::Layer<1, 0> g;

  const float learning_rate = 0.001;
  const unsigned int modulo_number = 100;
  const unsigned int learn_samples = 1000;
  const unsigned int tests = 1000000;
  const unsigned int epoch = 1000;
  const float tolerance = 1.0;
  e.setLearningRate(learning_rate);
  g.setLearningRate(learning_rate);
  fmt::println(fg(fmt::color::yellow), "-- Parameters:");
  fmt::println(fg(fmt::color::white), "learning_rate = {}", learning_rate);
  fmt::println(fg(fmt::color::white), "modulo_number = {}", modulo_number);
  fmt::println(fg(fmt::color::white), "learn_samples = {}", learn_samples);
  fmt::println(fg(fmt::color::white), "tests = {}", tests);
  fmt::println(fg(fmt::color::white), "epoch = {}", epoch);
  fmt::println(fg(fmt::color::white), "tolerance = {}", tolerance);

  fmt::println(fg(fmt::color::yellow), "-- Learning from random set");
  for(unsigned int j = 0; j < epoch; j++){
    for(unsigned int i = 0; i < learn_samples; i++) {
      float x = (rand()%modulo_number);
      float y = (rand()%modulo_number);

      NN::Utils::progressBar(i + j * learn_samples, learn_samples * epoch);

      e.setNodes({x / modulo_number, y / modulo_number});
      e.forward(g);

      float val = x - y;
      float res = (val / modulo_number);
      g.backprop_initial({res});
      e.backprop(g);
    };
  };

  fmt::println(fg(fmt::color::yellow), "-- Testing on random set");
  float sum = 0;
  for(unsigned int i = 0; i < tests; i++) {
    float x = (rand()%modulo_number);
    float y = (rand()%modulo_number);

    NN::Utils::progressBar(i, tests);

    e.setNodes({x / modulo_number, y / modulo_number});
    e.forward(g);

    float val = x - y;
    float res = val / modulo_number;
    if ((fabs(g[0] - res) * modulo_number < tolerance ? 1 : 0)) sum += 1;
  };
  if((sum / tests) * 100 >= 90) fmt::println(fg(fmt::color::green), "avg: {}%", (sum / tests) * 100);
  else                          fmt::println(fg(fmt::color::red), "avg: {}%", (sum / tests) * 100);


  std::string input = "";
  while(input != "q"){
    fmt::println(fg(fmt::color::light_golden_rod_yellow), "-- App --");
    fmt::println(fg(fmt::color::white), "-- q to exit");
    
    fmt::println(fg(fmt::color::white), "| | - b = ?");
    std::getline(std::cin, input);

    if(input == "q") break;

    int a = 0;
    try{
      a = std::stoi(input);
    }
    catch(...){
      fmt::println("Invalid integer");
      continue;
    };

    fmt::println(fg(fmt::color::white), "{} - | | = ?", a);
    std::getline(std::cin, input);
    int b = 0;
    try{
      b = std::stoi(input);
    }
    catch(...){
      fmt::println("Invalid integer");
      continue;
    };

    e.setNodes({(float)a / modulo_number, (float)b / modulo_number});
    e.forward(g);
    fmt::println(fg(fmt::color::violet), "{} - {} = {}", a, b, g[0] * modulo_number);
  };
};
};