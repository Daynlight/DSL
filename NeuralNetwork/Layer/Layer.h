// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <string>
#include <stdexcept>
#include <random>
#include <memory>
#include <cstring>
#include <span>
#include <cstdint>
#include <initializer_list>
#include <algorithm>

#include "../Activation/Activation.h"
#include "../Loss/Loss.h"
#include "../ThreadPool/ThreadPool.h"



namespace NN {
enum class Field : uint32_t{
  S_val = 1,
  D_val = 2,
  LEARNING_RATE = 3,
  ACTIVATION = 4,
  LOSS = 5,
  WEIGHTS = 6,
  END = 0
};

enum class Activation : uint32_t{
  LINEAR = 1,
  SIGMOID = 2,
  SOFTMAX = 3,
  RELU = 4
};

enum class Loss : uint32_t{
  MSE = 1,
  CROSSENTROPY = 2
};



class LayerWorkers {
protected:
  inline static ThreadPool workers{
    std::max(1u, std::thread::hardware_concurrency())
  };
};



template<unsigned int S, unsigned int D>
class Layer : private LayerWorkers{
//----------------------------------------------------------------//
//------------------------ Representation ------------------------//
//----------------------------------------------------------------//
// Nodes are S + 1 double Array where last one is always 1 for bias
// todo: will be changed to Math:Vec in feature
//----------------------------------------------------------------//
// Weights is matrix S + 1 Columns and D Rows, D is 1 less then 
// second layer nodes because last node is always 1 and we don't
// update it
// todo: will be change to Math::Matrix in feature
//----------------------------------------------------------------//
// Loss is interfaced to be function and prime function look Loss.h
// always setted default MSE f(x) = ((x - t)^2)/2, f'(x) = x - t
// todo: change from pointer to function type
//----------------------------------------------------------------//
// Activation is interfaced to be function and prime function
// look Activation.h always setted default Linear f(x) = x, f'(x) = 1
// todo: change from pointer to function type
//----------------------------------------------------------------//
// Learning rate indicate learning speed
//----------------------------------------------------------------//

// ======================================== //
// ================= Data ================= //
// ======================================== //
private:
  double nodes[S + 1];
  double sigma[S];
  double weights[(S + 1) * D];
  double activated[S];
  bool activated_after_forward = false;
  std::unique_ptr<NN::iActivation> activation = nullptr;
  std::unique_ptr<NN::iLoss> loss = nullptr;
  double learning_rate = 0.005;

// ======================================== //
// ============== Functions =============== //
// ======================================== //
// =========================== //
// ======= Constructors ====== //
// =========================== //
public:
  Layer() noexcept;   // O((S+1) * D)
  ~Layer() noexcept;  // O(1)
  
// ============================== //
// ======= Setters/Getters ====== //
// ============================== //
public:
  double* getNodes() noexcept;   // O(1)
  double* getActivatedNodes() noexcept;
  void setNodes(std::initializer_list<double> nodes) noexcept;    // O(n)
  void setNodes(std::span<const double> nodes) noexcept;    // O(n)
  double getActivatedNode(unsigned int i) const noexcept;    // O(1)
  void activateNodes() noexcept;

  double* getWeights() noexcept;   // O(1)
  void setWeights(std::initializer_list<double> weights) noexcept;    // O(n)
  void setWeights(const double* weights) noexcept;
  void setWeights(double min, double max) noexcept;

  double getLearningRate() const noexcept;    // O(1)
  void setLearningRate(double learning_rate) noexcept;    // O(1)

  const std::unique_ptr<iActivation>& getActivation() const noexcept;    // O(1)
  template<typename T>
  void setActivation() noexcept;   // O(1)

  const std::unique_ptr<iLoss>& getLoss() const noexcept;    // O(1)
  template<typename T>
  void setLoss() noexcept;   // O(1)

  double& operator[](unsigned int i);   // O(1)
  const double *getSigma() const noexcept;    // O(1)

// =============================== //
// ======= Forward/Backprop ====== //
// =============================== //
public:
  template<unsigned int N>
  void forward(Layer<D, N> &layer);   // O(n^2)

  void backprop_initial(std::initializer_list<double> target) noexcept;
  void backprop_initial(std::span<const double> target) noexcept;
  void backprop_initial_softmax_cross_entropy_fuse(std::span<const double> target) noexcept;
  template<unsigned int N>
  void backprop(Layer<D, N> &layer) noexcept;

// =========================== //
// ======= Presentation ====== //
// =========================== //
public:
  std::string print() const;    // O(n)
  std::string serialize() const noexcept;    // O(n)
  void deserialize(const std::string& data);   // O(n)
};
};



#include "Layer.hpp"
