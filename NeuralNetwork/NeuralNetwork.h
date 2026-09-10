// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once

#include "NeuralNetwork/Layer/Layer.h"
#include <tuple>
#include <utility>
#include <memory>
#include <span>



namespace NN {
template <unsigned int... Ns>
struct LayerBuilder;
template <unsigned int A, unsigned int B, unsigned int... Rest>
struct LayerBuilder<A, B, Rest...> {
  using type = decltype(std::tuple_cat(
    std::tuple<Layer<A, B>>{},
    typename LayerBuilder<B, Rest...>::type{}
  ));
};

template <unsigned int A, unsigned int B>
struct LayerBuilder<A, B> {
  using type = std::tuple<Layer<A, B>>;
};

template <std::size_t... I>
constexpr auto reverse_sequence(std::index_sequence<I...>) {
  return std::index_sequence<sizeof...(I) - 1 - I...>{};
};



template<unsigned int... L>
class NeuralNetwork;
template<unsigned int First, unsigned int Second, unsigned int... Rest>
class NeuralNetwork<First, Second, Rest...> {
// =================================== //
// ============== Data =============== //
// =================================== //
private:
  using LayerTuple = typename LayerBuilder<First, Second, Rest...>::type;
  LayerTuple layers;



// ======================================== //
// ============== Functions =============== //
// ======================================== //
// =========================== //
// ======= Constructors ====== //
// =========================== //
public:
  // core
  NeuralNetwork() noexcept;
  ~NeuralNetwork() noexcept;
  // copy
  NeuralNetwork(const NeuralNetwork& second) noexcept;
  NeuralNetwork& operator=(const NeuralNetwork& second) noexcept;
  // move
  NeuralNetwork(NeuralNetwork&& second) noexcept;
  NeuralNetwork& operator=(NeuralNetwork&& second) noexcept;

// =========================== //
// ===== Setters/Getters ===== //
// =========================== //
public:
  void setLearningRate(float learning_rate) noexcept;
  template<std::size_t I, typename T>
  void setActivation() noexcept;
  template<std::size_t I, typename T>
  void setLoss() noexcept;
  void setInput(std::initializer_list<float> nodes) noexcept;
  void setInput(std::span<const float> nodes) noexcept;
  void setWeights(float min, float max) noexcept;
  template<std::size_t I>
  void setWeights(float min, float max) noexcept;
  void setGPUAcceleration(bool value) noexcept;
  template<std::size_t I>
  void setGPUAcceleration(bool value) noexcept;
  float* getResult() noexcept;
  float* getActivatedResult() noexcept;
  const float* getInputSigma() noexcept;

// =========================== //
// ===== Forward/Backprop ==== //
// =========================== //
public:
  void forward();
  template <std::size_t... I>
  void forwardImpl(std::index_sequence<I...>);
  
  void backprop(std::initializer_list<float> loss);
  void backprop(std::span<const float> loss);
  template <std::size_t... I>
  void backpropImpl(std::index_sequence<I...>);
  void backpropInitial(std::initializer_list<float> loss);
  void backpropInitial(std::span<const float> loss);


// =========================== //
// ======= Presentation ====== //
// =========================== //
public:
  std::string print() const;    // O(n)
  std::string serialize() const noexcept;    // O(n)
  void deserialize(const std::string& data);   // O(n)
};
};



#include "NeuralNetwork.hpp"