# Neural Network
# Copyright 2026 Daynlight
# Licensed under the GNU General.
# See LICENSE file for details.



cmake_minimum_required(VERSION 3.15)

project(NeuralNetwork LANGUAGES CXX)

set(src
  ThreadPool/ThreadPool.h
  ThreadPool/ThreadPool.hpp
  Activation/Activation.h
  Activation/Activation.hpp
  Loss/Loss.h
  Loss/Loss.hpp
  Layer/Layer.h
  Layer/Layer.hpp
  NeuralNetwork.h
  NeuralNetwork.hpp
  NeuralNetwork.cpp
)

add_library(NeuralNetwork STATIC ${src})
add_library(NeuralNetwork::NeuralNetwork ALIAS NeuralNetwork)

target_include_directories(
  NeuralNetwork
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<INSTALL_INTERFACE:include>
)