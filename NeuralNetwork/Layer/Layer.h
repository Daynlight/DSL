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

#include "../GPUAcceleration/GPUAcceleration.h" 
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
// Nodes are S + 1 float Array where last one is always 1 for bias
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
template<unsigned int, unsigned int> friend class Layer;
private:
  enum GPUBuffer : size_t {
    GPUHeader = 0,
    GPUWeights = 1,
    GPUNodes = 2,
    GPUActivated = 3,
    GPUSigma = 4,
    GPUGradient = 5,
    GPUTarget = 6,
    GPUBufferCount = 7,
    GPUNext = 7
  };

  float nodes[S + 1];
  float sigma[S];
  float weights[(S + 1) * D];
  float activated[S];
  std::vector<std::vector<float>> partial_gradients;
  std::unique_ptr<NN::iActivation> activation = nullptr;
  std::unique_ptr<NN::iLoss> loss = nullptr;
  float learning_rate = 0.005;
  bool gpu_acceleration = true;
  const size_t gpu_acceleration_size_threshold = 65536;
  const size_t multithreading_acceleration_size_threshold = 4096;
  bool gpu_header_dirty = true;
  bool gpu_storage_ready = false;
  bool gpu_weights_dirty = true;
  bool gpu_nodes_dirty = true;
  bool gpu_activated_dirty = false;
  bool gpu_sigma_dirty = false;
  bool cpu_nodes_dirty = false;
  bool cpu_weights_dirty = false;
  bool cpu_activated_dirty = false;
  bool cpu_sigma_dirty = false;
  const bool debug_logs = false;
  
  bool activated_after_forward = false;
  std::vector<CW::Renderer::GPUStore> gpu_storage;
  

  
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
  float* getNodes() noexcept;   // O(1)
  void setNodes(std::initializer_list<float> nodes) noexcept;    // O(n)
  void setNodes(std::span<const float> nodes) noexcept;    // O(n)
  
  float* getActivatedNodes() noexcept;
  float getActivatedNode(unsigned int i) noexcept;
  void activateNodes_cpu() noexcept;
  void activateNodes_threads() noexcept;
  void activateNodes_gpu() noexcept;
  void activateNodes() noexcept;

  float* getWeights() noexcept;   // O(1)
  void setWeights(std::initializer_list<float> weights) noexcept;    // O(n)
  void setWeights(const float* weights) noexcept;
  void setWeights(float min, float max) noexcept;

  float getLearningRate() const noexcept;    // O(1)
  void setLearningRate(float learning_rate) noexcept;    // O(1)

  bool getGpuAcceleration() const noexcept;
  void setGpuAcceleration(bool value) noexcept;
  void ensure_gpu_storage();
  void sync_gpu_header() noexcept;

  const std::unique_ptr<iActivation>& getActivation() const noexcept;    // O(1)
  template<typename T>
  void setActivation() noexcept;   // O(1)

  const std::unique_ptr<iLoss>& getLoss() const noexcept;    // O(1)
  template<typename T>
  void setLoss() noexcept;   // O(1)

  float& operator[](unsigned int i);   // O(1)
  const float *getSigma() noexcept;    // O(1)

// =============================== //
// ======= Forward/Backprop ====== //
// =============================== //
public:
  template<unsigned int N> 
  void forward_cpu(Layer<D, N>& layer) noexcept;
  template<unsigned int N> 
  void forward_threads(Layer<D, N>& layer) noexcept;
  template<unsigned int N> 
  void forward_gpu(Layer<D, N>& layer);
  template<unsigned int N>
  void forward(Layer<D, N> &layer);   // O(n^2)

  void backprop_initial(std::initializer_list<float> target) noexcept;
  void backprop_initial_cpu(std::span<const float> target) noexcept;
  void backprop_initial_threads(std::span<const float> target) noexcept;
  void backprop_initial_gpu(std::span<const float> target) noexcept;
  void backprop_initial(std::span<const float> target) noexcept;

  void backprop_initial_softmax_cross_entropy_fuse_cpu(std::span<const float> target) noexcept;
  void backprop_initial_softmax_cross_entropy_fuse_threads(std::span<const float> target) noexcept;
  void backprop_initial_softmax_cross_entropy_fuse_gpu(std::span<const float> target) noexcept;
  void backprop_initial_softmax_cross_entropy_fuse(std::span<const float> target) noexcept;
  
  template<unsigned int N>
  void backprop_cpu(Layer<D, N> &layer) noexcept;
  template<unsigned int N>
  void backprop_threads(Layer<D, N> &layer) noexcept;
  template<unsigned int N>
  void backprop_gpu(Layer<D, N> &layer) noexcept;
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
