// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "NeuralNetwork.h"



// =========================== //
// ======= Constructors ====== //
// =========================== //
// core
template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>::NeuralNetwork() noexcept {};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>::~NeuralNetwork() noexcept {};



// copy
template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>::NeuralNetwork(const NeuralNetwork& second) noexcept
  : layers(second.layers) {};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>& NN::NeuralNetwork<First, Second, Rest...>::operator=(const NeuralNetwork& second) noexcept {
  if(this == &second) return *this;

  layers = second.layers;

  return *this;
};



// move
template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>::NeuralNetwork(NeuralNetwork&& second) noexcept
  : layers(std::move(second.layers)) {};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline NN::NeuralNetwork<First, Second, Rest...>& NN::NeuralNetwork<First, Second, Rest...>::operator=(NeuralNetwork&& second) noexcept {
  if(this == &second) return *this;

  layers = std::move(second.layers);

  return *this;
};



// =========================== //
// ======= Constructors ====== //
// =========================== //
template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::setLearningRate(float learning_rate) noexcept {
  std::apply([&](auto&... layer) {
    (layer.setLearningRate(learning_rate), ...);
  }, layers);
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
template<std::size_t I, typename T>
inline void NN::NeuralNetwork<First, Second, Rest...>::setActivation() noexcept {
  if constexpr(I < 0) return;
  if constexpr(I >= std::tuple_size_v<LayerTuple>) return;
  std::get<I>(layers).template setActivation<T>();
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
template<std::size_t I, typename T>
inline void NN::NeuralNetwork<First, Second, Rest...>::setLoss() noexcept {
  if constexpr(I < 0) return;
  if constexpr(I >= std::tuple_size_v<LayerTuple>) return;
  std::get<I>(layers).template setLoss<T>();
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::setInput(std::initializer_list<float> nodes) noexcept {
  std::get<0>(layers).setNodes(nodes);
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::setInput(std::span<const float> nodes) noexcept {
  std::get<0>(layers).setNodes(nodes);
};



template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::setWeights(float min, float max) noexcept {
  std::apply([&](auto&... layer) {
    (layer.setWeights(min, max), ...);
  }, layers);
};



template <unsigned int First, unsigned int Second, unsigned int... Rest>
template<std::size_t I>
inline void NN::NeuralNetwork<First, Second, Rest...>::setWeights(float min, float max) noexcept {
  if constexpr(I < 0) return;
  if constexpr(I >= std::tuple_size_v<LayerTuple>) return;
  std::get<I>(layers).setWeights(min, max);
};



template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::setGPUAcceleration(bool value) noexcept {
  std::apply([&](auto&... layer) {
    (layer.setGpuAcceleration(value), ...);
  }, layers);
};



template <unsigned int First, unsigned int Second, unsigned int... Rest>
template<std::size_t I>
inline void NN::NeuralNetwork<First, Second, Rest...>::setGPUAcceleration(bool value) noexcept {
  if constexpr(I < 0) return;
  if constexpr(I >= std::tuple_size_v<LayerTuple>) return;
  std::get<I>(layers).setGpuAcceleration(value);
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline float* NN::NeuralNetwork<First, Second, Rest...>::getResult() noexcept {
  if constexpr(std::tuple_size_v<LayerTuple> <= 0) return nullptr;
  constexpr std::size_t last = std::tuple_size_v<LayerTuple> - 1;
  return std::get<last>(layers).getNodes();
};



template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline float *NN::NeuralNetwork<First, Second, Rest...>::getActivatedResult() noexcept {
  if constexpr(std::tuple_size_v<LayerTuple> <= 0) return nullptr;
  constexpr std::size_t last = std::tuple_size_v<LayerTuple> - 1;
  return std::get<last>(layers).getActivatedNodes();
}



// =========================== //
// ===== Forward/Backprop ==== //
// =========================== //
template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::forward(){
  if constexpr(std::tuple_size_v<LayerTuple> <= 0) return;
  forwardImpl(std::make_index_sequence<std::tuple_size_v<LayerTuple> - 1>{});
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
template <std::size_t... I>
inline void NN::NeuralNetwork<First, Second, Rest...>::forwardImpl(std::index_sequence<I...>){
  ((std::get<I>(layers).forward(std::get<I + 1>(layers))), ...);
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::backprop(std::initializer_list<float> loss){
  if constexpr(std::tuple_size_v<LayerTuple> <= 0) return;
  backpropInitial(loss);
  backpropImpl(reverse_sequence(std::make_index_sequence<std::tuple_size_v<LayerTuple> - 1>{}));
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::backprop(std::span<const float> loss){
  if constexpr(std::tuple_size_v<LayerTuple> <= 0) return;
  backpropInitial(loss);
  backpropImpl(reverse_sequence(std::make_index_sequence<std::tuple_size_v<LayerTuple> - 1>{}));
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
template <std::size_t... I>
inline void NN::NeuralNetwork<First, Second, Rest...>::backpropImpl(std::index_sequence<I...>){
  ((std::get<I>(layers).backprop(std::get<I + 1>(layers))), ...);
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::backpropInitial(std::initializer_list<float> loss){
  if constexpr(std::tuple_size_v<LayerTuple> <= 0) return;
  constexpr std::size_t last = std::tuple_size_v<LayerTuple> - 1;
  std::get<last>(layers).backprop_initial(loss);
};



template<unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::backpropInitial(std::span<const float> loss){
  if constexpr(std::tuple_size_v<LayerTuple> <= 0) return;
  constexpr std::size_t last = std::tuple_size_v<LayerTuple> - 1;
  std::get<last>(layers).backprop_initial(loss);
};



template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline const  float *NN::NeuralNetwork<First, Second, Rest...>::getInputSigma() noexcept{
  return std::get<0>(layers).getSigma();
};



// =========================== //
// ======= Presentation ====== //
// =========================== //
template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline std::string NN::NeuralNetwork<First, Second, Rest...>::print() const{
  std::string s = "NeuralNetwork:\n";

  unsigned int i = 0;

  std::apply([&](const auto&... layer){
    ((s += "\nLayer " + std::to_string(i++) + ":\n" + layer.print()), ...);
  }, layers);

  return s;
};



template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline std::string NN::NeuralNetwork<First, Second, Rest...>::serialize() const noexcept{
  std::string data;
  uint32_t layer_count = std::tuple_size_v<LayerTuple>;

  data.append(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));

  std::apply([&](const auto&... layer){
    auto write_layer = [&](const auto& current_layer){
      std::string layer_data = current_layer.serialize();
      uint64_t layer_size = layer_data.size();

      data.append(reinterpret_cast<const char*>(&layer_size), sizeof(layer_size));

      if(layer_size > 0) data.append(layer_data.data(), layer_size);
    };
    (write_layer(layer), ...);
  }, layers);
  return data;
};



template <unsigned int First, unsigned int Second, unsigned int... Rest>
inline void NN::NeuralNetwork<First, Second, Rest...>::deserialize(const std::string& data){
  size_t offset = 0;

  auto read = [&](void* destination, size_t size){
    if(offset + size > data.size()) throw std::runtime_error("Invalid serialized NeuralNetwork data");

    std::memcpy(destination, data.data() + offset, size);
    offset += size;
  };

  uint32_t layer_count = 0;
  read(&layer_count, sizeof(layer_count));

  if(layer_count != std::tuple_size_v<LayerTuple>) throw std::runtime_error("NeuralNetwork layer count mismatch");

  std::apply([&](auto&... layer){
    auto read_layer = [&](auto& current_layer){
      uint64_t layer_size = 0;
      read(&layer_size, sizeof(layer_size));

      if(offset + layer_size > data.size()) throw std::runtime_error("Invalid serialized NeuralNetwork layer size");

      std::string layer_data(data.data() + offset, layer_size);

      offset += layer_size;
      current_layer.deserialize(layer_data);
    };
    (read_layer(layer), ...);
  }, layers);
};