// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "Layer.h"



// =========================== //
// ======= Constructors ====== //
// =========================== //
// core
template <unsigned int S, unsigned int D>
inline NN::Layer<S, D>::Layer() noexcept {
  NN::GPUAcceleration::get();
  nodes[S] = 1.0f;
  setWeights(-1.0f, 1.0f);
  loss = std::make_unique<NN::MSE>();
  activation = std::make_unique<NN::Linear>();
};



template <unsigned int S, unsigned int D>
inline NN::Layer<S, D>::~Layer() noexcept { };



// ============================== //
// ======= Setters/Getters ====== //
// ============================== //
template <unsigned int S, unsigned int D>
inline float *NN::Layer<S, D>::getNodes() noexcept {
  glFinish();
  if(cpu_nodes_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUNodes].get(data);
    std::memcpy(nodes, data.data(), sizeof(nodes));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("getNodes GPUNodes Get: %zu bytes, %.3f ms\n", sizeof(nodes), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_nodes_dirty = false;
  };

  gpu_nodes_dirty = true;
  activated_after_forward = false;

  glFinish();
  return nodes;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setNodes(std::initializer_list<float> nodes) noexcept {
  glFinish();
  const size_t size = std::min(nodes.size(), size_t(S));

  std::copy_n(nodes.begin(), size, this->nodes);
  std::fill(this->nodes + size, this->nodes + S, 0.0f);

  this->nodes[S] = 1.0f;
  gpu_nodes_dirty = true;
  cpu_nodes_dirty = false;
  activated_after_forward = false;
  glFinish();
};



template<unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setNodes(std::span<const float> nodes) noexcept {
  glFinish();
  const size_t size = std::min(nodes.size(), size_t(S));

  std::copy_n(nodes.begin(), size, this->nodes);
  std::fill(this->nodes + size, this->nodes + S, 0.0f);

  this->nodes[S] = 1.0f;
  gpu_nodes_dirty = true;
  cpu_nodes_dirty = false;
  activated_after_forward = false;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline float *NN::Layer<S, D>::getActivatedNodes() noexcept {
  glFinish();
  activateNodes();

  if(cpu_activated_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUActivated].get(data);
    std::memcpy(activated, data.data(), sizeof(activated));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("getActivatedNodes GPUActivated Get: %zu bytes, %.3f ms\n", sizeof(activated), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_activated_dirty = false;
  };

  glFinish();
  return activated;
};



template <unsigned int S, unsigned int D>
inline float NN::Layer<S, D>::getActivatedNode(unsigned int i) noexcept {
  glFinish();
  if(cpu_nodes_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUNodes].get(data);
    std::memcpy(nodes, data.data(), sizeof(nodes));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("getActivatedNode GPUNodes Get: %zu bytes, %.3f ms\n", sizeof(nodes), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_nodes_dirty = false;
  };

  std::span<const float> layer(nodes, S);
  glFinish();
  return activation->fun(layer, i);
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::activateNodes_cpu() noexcept {
  glFinish();
  if(cpu_nodes_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUNodes].get(data);
    std::memcpy(nodes, data.data(), sizeof(nodes));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("activateNodes_cpu GPUNodes Get: %zu bytes, %.3f ms\n", sizeof(nodes), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_nodes_dirty = false;
  };

  for(size_t i = 0; i < S; i++) activated[i] = getActivatedNode(i);

  cpu_activated_dirty = false;
  gpu_activated_dirty = true;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::activateNodes_threads() noexcept {
  glFinish();
  if(cpu_nodes_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUNodes].get(data);
    std::memcpy(nodes, data.data(), sizeof(nodes));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("activateNodes_threads GPUNodes Get: %zu bytes, %.3f ms\n", sizeof(nodes), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_nodes_dirty = false;
  };

  workers.parallel_for(0, S, [&](size_t begin, size_t end){
    for(size_t i = begin; i < end; i++) activated[i] = getActivatedNode(i);
  });

  workers.wait();

  cpu_activated_dirty = false;
  gpu_activated_dirty = true;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::activateNodes_gpu() noexcept {
  glFinish();
  ensure_gpu_storage();

  auto& gpu = NN::GPUAcceleration::get();
  CW::Renderer::ComputeShader* shader = nullptr;
  unsigned int groups = (S + 255) / 256;

  if(dynamic_cast<NN::Linear*>(activation.get())) shader = &gpu.getActivationLinearShader();
  else if(dynamic_cast<NN::Sigmoid*>(activation.get())) shader = &gpu.getActivationSigmoidShader();
  else if(dynamic_cast<NN::ReLU*>(activation.get())) shader = &gpu.getActivationReLUShader();
  else if(dynamic_cast<NN::Softmax*>(activation.get())){
    shader = &gpu.getActivationSoftmaxShader();
    groups = 1;
  };

  if(!shader) return;

  if(debug_logs){
    glFinish();
  };
  auto start = std::chrono::steady_clock::now();
  shader->run(gpu_storage, groups);
  if(debug_logs){
    glFinish();
    auto end = std::chrono::steady_clock::now();
    printf("activateNodes_gpu ActivateNodesShader Run: %.3f ms\n", std::chrono::duration<double, std::milli>(end - start).count());
  };
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glFinish();
  for(GLenum error; (error = glGetError()) != GL_NO_ERROR;){
    printf("Activation OpenGL error: 0x%X\n", error);
  }

  cpu_activated_dirty = true;
  gpu_activated_dirty = false;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::activateNodes() noexcept {
  glFinish();
  if(activated_after_forward) return;

  if(!gpu_acceleration){
    if(cpu_nodes_dirty){
      std::vector<float> data;
      if(debug_logs){
        glFinish();
      };
      auto start = std::chrono::steady_clock::now();
      gpu_storage[GPUNodes].get(data);
      std::memcpy(nodes, data.data(), sizeof(nodes));
      if(debug_logs){
        glFinish();
        auto end = std::chrono::steady_clock::now();
        printf("activateNodes GPUNodes Get: %zu bytes, %.3f ms\n", sizeof(nodes), std::chrono::duration<double, std::milli>(end - start).count());
      };
      cpu_nodes_dirty = false;
    };

    if(size_t(S) > multithreading_acceleration_size_threshold) activateNodes_threads();
    else activateNodes_cpu();

    cpu_activated_dirty = false;
    gpu_activated_dirty = true;
  }
  else{
    activateNodes_gpu();

    cpu_activated_dirty = true;
    gpu_activated_dirty = false;
  };

  activated_after_forward = true;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline float *NN::Layer<S, D>::getWeights() noexcept {
  glFinish();
  if(cpu_weights_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUWeights].get(data);
    std::memcpy(weights, data.data(), sizeof(weights));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("getWeights GPUWeights Get: %zu bytes, %.3f ms\n", sizeof(data), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_weights_dirty = false;
  };

  gpu_weights_dirty = true;
  glFinish();
  return weights;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setWeights(std::initializer_list<float> weights) noexcept {
  glFinish();
  const size_t size = std::min(weights.size(), size_t((S + 1) * D));

  std::copy_n(weights.begin(), size, this->weights);
  std::fill(this->weights + size, this->weights + (S + 1) * D, 0.0f);

  gpu_weights_dirty = true;
  cpu_weights_dirty = false;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setWeights(const float* weights) noexcept {
  glFinish();
  std::copy_n(weights, (S + 1) * D, this->weights);

  gpu_weights_dirty = true;
  cpu_weights_dirty = false;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setWeights(float min, float max) noexcept {  
  glFinish();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min, max);
  
  for(float &el : weights) el = dist(gen);
  for(size_t j = 0; j < D; j++) weights[j * (S + 1) + S] = 0.0;
  
  gpu_weights_dirty = true;
  cpu_weights_dirty = false;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline float NN::Layer<S, D>::getLearningRate() const noexcept {
  return learning_rate;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setLearningRate(float learning_rate) noexcept {
  glFinish();
  this->learning_rate = learning_rate;
  gpu_header_dirty = true;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline bool NN::Layer<S, D>::getGpuAcceleration() const noexcept {
  return gpu_acceleration;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setGpuAcceleration(bool value) noexcept {
  glFinish();
  gpu_acceleration = value;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::ensure_gpu_storage() {
  glFinish();
  if(gpu_weights_dirty && cpu_weights_dirty)
    throw std::runtime_error("GPU and CPU weights are both dirty");

  if(gpu_nodes_dirty && cpu_nodes_dirty)
    throw std::runtime_error("GPU and CPU nodes are both dirty");

  if(gpu_activated_dirty && cpu_activated_dirty)
    throw std::runtime_error("GPU and CPU activated are both dirty");

  if(gpu_sigma_dirty && cpu_sigma_dirty)
    throw std::runtime_error("GPU and CPU sigma are both dirty");

  if(!gpu_storage_ready){
    gpu_storage.resize(GPUBufferCount);

    nodes[S] = 1.0f;
    NN::GPUAcceleration::get();

    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUNodes].set(std::vector<float>(nodes, nodes + S + 1));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("Ensure Initial GPUNodes Set: %zu bytes, %.3f ms\n", sizeof(nodes), std::chrono::duration<double, std::milli>(end - start).count());
    };

    if(debug_logs){
      glFinish();
    };
    start = std::chrono::steady_clock::now();
    gpu_storage[GPUActivated].set(std::vector<float>(S, 0.0f));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("Ensure Initial GPUActivated Set: %zu bytes, %.3f ms\n", sizeof(float) * S, std::chrono::duration<double, std::milli>(end - start).count());
    };

    if(debug_logs){
      glFinish();
    };
    start = std::chrono::steady_clock::now();
    gpu_storage[GPUSigma].set(std::vector<float>(S, 0.0f));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("Ensure Initial GPUSigma Set: %zu bytes, %.3f ms\n", sizeof(float) * S, std::chrono::duration<double, std::milli>(end - start).count());
    };

    if(debug_logs){
      glFinish();
    };
    start = std::chrono::steady_clock::now();
    gpu_storage[GPUGradient].set(std::vector<float>(S, 0.0f));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("Ensure Initial GPUGradient Set: %zu bytes, %.3f ms\n", sizeof(float) * S, std::chrono::duration<double, std::milli>(end - start).count());
    };
    
    if(debug_logs){
      glFinish();
    };
    start = std::chrono::steady_clock::now();
    gpu_storage[GPUTarget].set(std::vector<float>(S, 0.0f));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("Ensure Initial GPUTarget Set: %zu bytes, %.3f ms\n", sizeof(float) * S, std::chrono::duration<double, std::milli>(end - start).count());
    };

    gpu_nodes_dirty = false;
    gpu_storage_ready = true;
  };

  if(gpu_weights_dirty){
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUWeights].set(std::vector<float>(weights, weights + (S + 1) * D));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("Ensure GPUWeights Set: %zu bytes, %.3f ms\n", sizeof(float) * S, std::chrono::duration<double, std::milli>(end - start).count());
    };
    gpu_weights_dirty = false;
  };

  if(gpu_nodes_dirty){
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUNodes].set(std::vector<float>(nodes, nodes + S + 1));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("Ensure GPUNodes Set: %zu bytes, %.3f ms\n", sizeof(float) * S, std::chrono::duration<double, std::milli>(end - start).count());
    };
    gpu_nodes_dirty = false;
  };

  if(gpu_activated_dirty){
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUActivated].set(std::vector<float>(activated, activated + S));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("Ensure GPUActivated Set: %zu bytes, %.3f ms\n", sizeof(float) * S, std::chrono::duration<double, std::milli>(end - start).count());
    };
    gpu_activated_dirty = false;
  };

  if(gpu_sigma_dirty){
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUSigma].set(std::vector<float>(sigma, sigma + S));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("Ensure GPUSigma Set: %zu bytes, %.3f ms\n", sizeof(float) * S, std::chrono::duration<double, std::milli>(end - start).count());
    };
    gpu_sigma_dirty = false;
  };

  glFinish();

  sync_gpu_header();
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::sync_gpu_header() noexcept {
  glFinish();
  if(!gpu_storage_ready || !gpu_header_dirty) return;

  unsigned int activation_id = 0;
  unsigned int loss_id = 0;

  if(dynamic_cast<NN::Linear*>(activation.get())) activation_id = 1;
  else if(dynamic_cast<NN::Sigmoid*>(activation.get())) activation_id = 2;
  else if(dynamic_cast<NN::Softmax*>(activation.get())) activation_id = 3;
  else if(dynamic_cast<NN::ReLU*>(activation.get())) activation_id = 4;

  if(dynamic_cast<NN::MSE*>(loss.get())) loss_id = 1;
  else if(dynamic_cast<NN::CrossEntropy*>(loss.get())) loss_id = 2;

  if(debug_logs){
    glFinish();
  };
  auto start = std::chrono::steady_clock::now();
  gpu_storage[GPUHeader].set(std::vector<float>{
    static_cast<float>(S),
    static_cast<float>(D),
    static_cast<float>(activation_id),
    static_cast<float>(loss_id),
    learning_rate
  });
  if(debug_logs){
    glFinish();
    auto end = std::chrono::steady_clock::now();
    printf("Sync GPUHeader Set: %zu bytes, %.3f ms\n", sizeof(float) * 5, std::chrono::duration<double, std::milli>(end - start).count());
  };

  glFinish();
  gpu_header_dirty = false;
};



template <unsigned int S, unsigned int D>
inline const std::unique_ptr<NN::iActivation>& NN::Layer<S, D>::getActivation() const noexcept {
  return activation;
};



template <unsigned int S, unsigned int D>
template<typename T>
inline void NN::Layer<S, D>::setActivation() noexcept {
  glFinish();
  activation = std::make_unique<T>();
  gpu_header_dirty = true;
  activated_after_forward = false;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline const std::unique_ptr<NN::iLoss>& NN::Layer<S, D>::getLoss() const noexcept {
  return loss;
};



template <unsigned int S, unsigned int D>
template<typename T>
inline void NN::Layer<S, D>::setLoss() noexcept {
  glFinish();
  loss = std::make_unique<T>();
  gpu_header_dirty = true;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline float &NN::Layer<S, D>::operator[](unsigned int i) {
  if(i > S) throw std::range_error("index out of range");
  return nodes[i];
};



template <unsigned int S, unsigned int D>
inline const float *NN::Layer<S, D>::getSigma() noexcept {
  glFinish();
  if(cpu_sigma_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUSigma].get(data);
    std::memcpy(sigma, data.data(), sizeof(sigma));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("getSigma GPUSigma Get: %zu bytes, %.3f ms\n", sizeof(sigma) * 5, std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_sigma_dirty = false;
  };
  
  glFinish();
  return sigma;
};



// =============================== //
// ======= Forward/Backprop ====== //
// =============================== //
template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::forward_cpu(Layer<D, N>& layer) noexcept {
  glFinish();
  if(cpu_weights_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUWeights].get(data);
    std::memcpy(weights, data.data(), sizeof(weights));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("forward_cpu GPUWeights Get: %zu bytes, %.3f ms\n", sizeof(weights), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_weights_dirty = false;
  };

  if(cpu_activated_dirty){
    std::vector<float> data;    
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUActivated].get(data);
    std::memcpy(activated, data.data(), sizeof(activated));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("forward_cpu GPUActivated Get: %zu bytes, %.3f ms\n", sizeof(activated), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_activated_dirty = false;
  };

  float output[D];

  for(size_t i = 0; i < D; i++){
    float sum = weights[i * (S + 1) + S];
    for(size_t j = 0; j < S; j++) sum += activated[j] * weights[i * (S + 1) + j];
    output[i] = sum;
  };

  std::memcpy(layer.getNodes(), output, sizeof(float) * D);
  layer[D] = 1.0f;

  layer.gpu_nodes_dirty = true;
  layer.cpu_nodes_dirty = false;
  layer.activated_after_forward = false;
  glFinish();
};



template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::forward_threads(Layer<D, N>& layer) noexcept {
  glFinish();
  if(cpu_weights_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUWeights].get(data);
    std::memcpy(weights, data.data(), sizeof(weights));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("forward_threads GPUWeights Get: %zu bytes, %.3f ms\n", sizeof(data), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_weights_dirty = false;
  };

  if(cpu_activated_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUActivated].get(data);
    std::memcpy(activated, data.data(), sizeof(activated));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("forward_threads GPUActivated Get: %zu bytes, %.3f ms\n", sizeof(activated), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_activated_dirty = false;
  };

  float output[D];

  workers.parallel_for(0, D, [&](size_t begin, size_t end){
    for(size_t i = begin; i < end; i++){
      float sum = weights[i * (S + 1) + S];
      for(size_t j = 0; j < S; j++) sum += activated[j] * weights[i * (S + 1) + j];
      output[i] = sum;
    };
  });

  workers.wait();

  std::memcpy(layer.getNodes(), output, sizeof(float) * D);
  layer[D] = 1.0f;

  layer.gpu_nodes_dirty = true;
  layer.cpu_nodes_dirty = false;
  layer.activated_after_forward = false;
  glFinish();
};



template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::forward_gpu(Layer<D, N>& layer) {
  glFinish();
  ensure_gpu_storage();
  layer.ensure_gpu_storage();

  layer.gpu_storage[GPUNodes].bind(GPUBufferCount);

  auto& shader = NN::GPUAcceleration::get().getForwardShader();
  
  glFinish();
  auto start = std::chrono::steady_clock::now();
  shader.run(gpu_storage, D);
  
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glFinish();
  for(GLenum error; (error = glGetError()) != GL_NO_ERROR;){
    printf("Backprop Forward Shader error: 0x%X\n", error);
  }
  if(debug_logs){
    auto end = std::chrono::steady_clock::now();
    printf("forward_gpu ForwardShader Run: %.3f ms\n", std::chrono::duration<double, std::milli>(end - start).count());
  };

  glFinish();

  layer.cpu_nodes_dirty = true;
  layer.gpu_nodes_dirty = false;
  layer.activated_after_forward = false;
};



template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::forward(NN::Layer<D, N>& layer) {
  glFinish();
  nodes[S] = 1.0;
  activated_after_forward = false;
  activateNodes();

  const size_t work = size_t(S) * D;

  if(!gpu_acceleration){
    if(work > multithreading_acceleration_size_threshold) forward_threads(layer);
    else forward_cpu(layer);
  }
  else{
    forward_gpu(layer);
  };
  glFinish();
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial(std::initializer_list<float> target) noexcept {
  backprop_initial(std::span<const float>(target.begin(), target.size()));
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial_cpu(std::span<const float> target) noexcept {
  glFinish();
  if(cpu_nodes_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUNodes].get(data);
    std::memcpy(nodes, data.data(), sizeof(nodes));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("backprop_initial_cpu GPUNodes Get: %zu bytes, %.3f ms\n", sizeof(nodes), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_nodes_dirty = false;
  };

  if(cpu_activated_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUActivated].get(data);
    std::memcpy(activated, data.data(), sizeof(activated));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("backprop_initial_cpu GPUActivated Get: %zu bytes, %.3f ms\n", sizeof(activated), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_activated_dirty = false;
  };

  const size_t size = std::min<size_t>(S, target.size());
  float output[S]{};
  std::span<const float> layer(nodes, S);
  float loss_gradient[S]{};

  for(size_t i = 0; i < size; i++) loss_gradient[i] = loss->fun_prime(activated[i], target[i]);

  for(size_t i = 0; i < S; i++){
    float sum = 0.0;
    for(size_t j = 0; j < size; j++) sum += loss_gradient[j] * activation->fun_prime(layer, j, i);
    output[i] = sum;
  };

  std::memcpy(sigma, output, sizeof(output));

  glFinish();
  cpu_sigma_dirty = false;
  gpu_sigma_dirty = true;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial_threads(std::span<const float> target) noexcept {
  glFinish();
  if(cpu_nodes_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUNodes].get(data);
    std::memcpy(nodes, data.data(), sizeof(nodes));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("backprop_initial_threads GPUNodes Get: %zu bytes, %.3f ms\n", sizeof(nodes), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_nodes_dirty = false;
  };

  if(cpu_activated_dirty){
    std::vector<float> data;
    if(debug_logs){
      glFinish();
    };
    auto start = std::chrono::steady_clock::now();
    gpu_storage[GPUActivated].get(data);
    std::memcpy(activated, data.data(), sizeof(activated));
    if(debug_logs){
      glFinish();
      auto end = std::chrono::steady_clock::now();
      printf("backprop_initial_threads GPUActivated Get: %zu bytes, %.3f ms\n", sizeof(activated), std::chrono::duration<double, std::milli>(end - start).count());
    };
    cpu_activated_dirty = false;
  };

  const size_t size = std::min<size_t>(S, target.size());
  float output[S]{};
  std::span<const float> layer(nodes, S);
  float loss_gradient[S]{};

  for(size_t i = 0; i < size; i++) loss_gradient[i] = loss->fun_prime(activated[i], target[i]);

  workers.parallel_for(0, S, [&](size_t begin, size_t end){
    for(size_t i = begin; i < end; i++){
      float sum = 0.0;
      for(size_t j = 0; j < size; j++) sum += loss_gradient[j] * activation->fun_prime(layer, j, i);
      output[i] = sum;
    };
  });

  workers.wait();

  std::memcpy(sigma, output, sizeof(output));

  cpu_sigma_dirty = false;
  gpu_sigma_dirty = true;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial_gpu(std::span<const float> target) noexcept {
  glFinish();
  unsigned int activation_id = 0;
  unsigned int loss_id = 0;

  if(dynamic_cast<NN::Linear*>(activation.get())) activation_id = 1;
  else if(dynamic_cast<NN::Sigmoid*>(activation.get())) activation_id = 2;
  else if(dynamic_cast<NN::Softmax*>(activation.get())) activation_id = 3;
  else if(dynamic_cast<NN::ReLU*>(activation.get())) activation_id = 4;

  if(dynamic_cast<NN::MSE*>(loss.get())) loss_id = 1;
  else if(dynamic_cast<NN::CrossEntropy*>(loss.get())) loss_id = 2;

  if(activation_id == 0 || loss_id == 0) return;
  ensure_gpu_storage();

  const size_t size = std::min<size_t>(S, target.size());
  std::vector<float> target_data(S, 0.0f);
  std::copy_n(target.data(), size, target_data.data());

  glFinish();
  auto start = std::chrono::steady_clock::now();
  gpu_storage[GPUTarget].set(target_data);
  glFinish();
  if(debug_logs){
    auto end = std::chrono::steady_clock::now();
    printf("backprop_initial_gpu GPUTarget Get: %zu bytes, %.3f ms\n", sizeof(target_data), std::chrono::duration<double, std::milli>(end - start).count());
  };

  auto& shader = NN::GPUAcceleration::get().getBackpropInitialShader();
  
  glFinish();
  start = std::chrono::steady_clock::now();
  shader.run(gpu_storage, activation_id == 3 ? 1 : (S + 255) / 256);

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glFinish();
  for(GLenum error; (error = glGetError()) != GL_NO_ERROR;){
    printf("Backprop Initial OpenGL error: 0x%X\n", error);
  }
  if(debug_logs){
    auto end = std::chrono::steady_clock::now();
    printf("backprop_initial_gpu BackpropInitial Run: %.3f ms\n", std::chrono::duration<double, std::milli>(end - start).count());
  };

  glFinish();
  cpu_sigma_dirty = true;
  gpu_sigma_dirty = false;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial(std::span<const float> target) noexcept {
  glFinish();
  activateNodes();
  if(dynamic_cast<NN::Softmax*>(activation.get()) && dynamic_cast<NN::CrossEntropy*>(loss.get())){
    backprop_initial_softmax_cross_entropy_fuse(target);
    return;
  };
  const size_t work = size_t(S) * std::min<size_t>(S, target.size());
  const bool supported_activation = dynamic_cast<NN::Linear*>(activation.get()) ||
                                    dynamic_cast<NN::Sigmoid*>(activation.get()) ||
                                    dynamic_cast<NN::ReLU*>(activation.get()) ||
                                    dynamic_cast<NN::Softmax*>(activation.get());
  const bool supported_loss = dynamic_cast<NN::MSE*>(loss.get()) ||
                              dynamic_cast<NN::CrossEntropy*>(loss.get());
  if(!gpu_acceleration || !supported_activation || !supported_loss){
    if(work > multithreading_acceleration_size_threshold) backprop_initial_threads(target);
    else backprop_initial_cpu(target);
  }
  else{
    backprop_initial_gpu(target);
  };
  glFinish();
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial_softmax_cross_entropy_fuse_cpu(std::span<const float> target) noexcept {
  glFinish();
  if(cpu_nodes_dirty){
    std::vector<float> data;
    gpu_storage[GPUNodes].get(data);
    std::memcpy(nodes, data.data(), sizeof(nodes));
    cpu_nodes_dirty = false;
  };

  float output[S]{};
  float exponent[S];
  float max = nodes[0];

  for(size_t i = 1; i < S; i++) if(nodes[i] > max) max = nodes[i];

  float sum = 0.0;

  for(size_t i = 0; i < S; i++){
    exponent[i] = std::exp(nodes[i] - max);
    sum += exponent[i];
  };

  const size_t size = std::min<size_t>(S, target.size());

  for(size_t i = 0; i < size; i++) output[i] = exponent[i] / sum - target[i];

  std::memcpy(sigma, output, sizeof(output));

  cpu_sigma_dirty = false;
  gpu_sigma_dirty = true;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial_softmax_cross_entropy_fuse_threads(std::span<const float> target) noexcept {
  glFinish();
  if(cpu_nodes_dirty){
    std::vector<float> data;
    gpu_storage[GPUNodes].get(data);
    std::memcpy(nodes, data.data(), sizeof(nodes));
    cpu_nodes_dirty = false;
  };

  float output[S]{};
  float exponent[S];
  float max = nodes[0];

  for(size_t i = 1; i < S; i++) if(nodes[i] > max) max = nodes[i];

  float sum = 0.0;

  for(size_t i = 0; i < S; i++){
    exponent[i] = std::exp(nodes[i] - max);
    sum += exponent[i];
  };

  const size_t size = std::min<size_t>(S, target.size());

  workers.parallel_for(0, size, [&](size_t begin, size_t end){
    for(size_t i = begin; i < end; i++) output[i] = exponent[i] / sum - target[i];
  });

  workers.wait();

  std::memcpy(sigma, output, sizeof(output));

  cpu_sigma_dirty = false;
  gpu_sigma_dirty = true;
  glFinish();
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial_softmax_cross_entropy_fuse_gpu(std::span<const float> target) noexcept {
  glFinish();
  ensure_gpu_storage();

  const size_t size = std::min<size_t>(S, target.size());
  std::vector<float> target_data(S, 0.0f);
  std::copy_n(target.data(), size, target_data.data());

  glFinish();
  auto start = std::chrono::steady_clock::now();
  gpu_storage[GPUTarget].set(target_data);
  glFinish();
  if(debug_logs){
    auto end = std::chrono::steady_clock::now();
    printf("backprop_initial_gpu GPUTarget Get: %zu bytes, %.3f ms\n", sizeof(target_data), std::chrono::duration<double, std::milli>(end - start).count());
  };

  auto& shader = NN::GPUAcceleration::get().getBackpropSoftmaxCrossEntropyShader();
  
  glFinish();
  start = std::chrono::steady_clock::now();
  shader.run(gpu_storage, (S + 255) / 256);
  
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glFinish();
  for(GLenum error; (error = glGetError()) != GL_NO_ERROR;){
    printf("Backprop Initial Softmax Crosss Entropy OpenGL error: 0x%X\n", error);
  }
  if(debug_logs){
    auto end = std::chrono::steady_clock::now();
    printf("backprop_initial_softmax_cross_entropy_fuse_gpu BackpropInitialSoftmaxEntropyFuse Run: %.3f ms\n", std::chrono::duration<double, std::milli>(end - start).count());
  };

  glFinish();

  cpu_sigma_dirty = true;
  gpu_sigma_dirty = false;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial_softmax_cross_entropy_fuse(std::span<const float> target) noexcept {
  glFinish();
  activateNodes();
  if(!gpu_acceleration){
    if(size_t(S) > multithreading_acceleration_size_threshold) backprop_initial_softmax_cross_entropy_fuse_threads(target);
    else backprop_initial_softmax_cross_entropy_fuse_cpu(target);
  }
  else{
    backprop_initial_softmax_cross_entropy_fuse_gpu(target);
  };
  activated_after_forward = false;
  glFinish();
};



template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::backprop_cpu(Layer<D, N>& next_layer) noexcept {
  glFinish();
  if(cpu_weights_dirty){
    std::vector<float> data;
    gpu_storage[GPUWeights].get(data);
    std::memcpy(weights, data.data(), sizeof(weights));
    cpu_weights_dirty = false;
  };

  if(cpu_nodes_dirty){
    std::vector<float> data;
    gpu_storage[GPUNodes].get(data);
    std::memcpy(nodes, data.data(), sizeof(nodes));
    cpu_nodes_dirty = false;
  };

  if(cpu_activated_dirty){
    std::vector<float> data;
    gpu_storage[GPUActivated].get(data);
    std::memcpy(activated, data.data(), sizeof(activated));
    cpu_activated_dirty = false;
  };

  const float* sigma_next = next_layer.getSigma();
  float gradient[S]{};
  float output[S]{};

  for(size_t j = 0; j < D; j++){
    const float factor = sigma_next[j];
    const float* row = weights + j * (S + 1);
    for(size_t i = 0; i < S; i++) gradient[i] += row[i] * factor;
  };

  std::span<const float> layer(nodes, S);

  if(dynamic_cast<NN::Linear*>(activation.get())){
    std::memcpy(output, gradient, sizeof(output));
  }
  else if(dynamic_cast<NN::Sigmoid*>(activation.get()) || dynamic_cast<NN::ReLU*>(activation.get())){
    for(size_t i = 0; i < S; i++) output[i] = gradient[i] * activation->fun_prime(layer, i, i);
  }
  else{
    for(size_t i = 0; i < S; i++){
      float sum = 0.0;
      for(size_t j = 0; j < S; j++) sum += gradient[j] * activation->fun_prime(layer, j, i);
      output[i] = sum;
    };
  };

  std::memcpy(sigma, output, sizeof(output));
  cpu_sigma_dirty = false;
  gpu_sigma_dirty = true;

  for(size_t j = 0; j < D; j++){
    const float factor = learning_rate * sigma_next[j];
    float* row = weights + j * (S + 1);
    for(size_t i = 0; i < S; i++) row[i] -= factor * activated[i];
    row[S] -= factor;
  };

  gpu_weights_dirty = true;
  cpu_weights_dirty = false;
  glFinish();
};



template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::backprop_threads(Layer<D, N>& next_layer) noexcept {
  glFinish();
  if(cpu_weights_dirty){
    std::vector<float> data;
    gpu_storage[GPUWeights].get(data);
    std::memcpy(weights, data.data(), sizeof(weights));
    cpu_weights_dirty = false;
  };

  if(cpu_nodes_dirty){
    std::vector<float> data;
    gpu_storage[GPUNodes].get(data);
    std::memcpy(nodes, data.data(), sizeof(nodes));
    cpu_nodes_dirty = false;
  };

  if(cpu_activated_dirty){
    std::vector<float> data;
    gpu_storage[GPUActivated].get(data);
    std::memcpy(activated, data.data(), sizeof(activated));
    cpu_activated_dirty = false;
  };

  const float* sigma_next = next_layer.getSigma();
  float gradient[S]{};
  float output[S]{};

  const size_t worker_count = std::min<size_t>(workers.size(), D);
  if(partial_gradients.size() != worker_count) partial_gradients.resize(worker_count);

  for(size_t worker = 0; worker < worker_count; worker++){
    if(partial_gradients[worker].size() != S) partial_gradients[worker].resize(S);
    std::fill(partial_gradients[worker].begin(), partial_gradients[worker].end(), 0.0);
  };

  workers.parallel_for(0, worker_count, [&](size_t begin, size_t end){
    for(size_t worker = begin; worker < end; worker++){
      float* partial = partial_gradients[worker].data();
      const size_t row_begin = worker * D / worker_count;
      const size_t row_end = (worker + 1) * D / worker_count;

      for(size_t j = row_begin; j < row_end; j++){
        const float factor = sigma_next[j];
        const float* row = weights + j * (S + 1);
        for(size_t i = 0; i < S; i++) partial[i] += row[i] * factor;
      };
    };
  });

  workers.wait();

  for(size_t worker = 0; worker < worker_count; worker++){
    const float* partial = partial_gradients[worker].data();
    for(size_t i = 0; i < S; i++) gradient[i] += partial[i];
  };

  std::span<const float> layer(nodes, S);

  if(dynamic_cast<NN::Linear*>(activation.get())){
    std::memcpy(output, gradient, sizeof(output));
  }
  else if(dynamic_cast<NN::Sigmoid*>(activation.get()) || dynamic_cast<NN::ReLU*>(activation.get())){
    for(size_t i = 0; i < S; i++) output[i] = gradient[i] * activation->fun_prime(layer, i, i);
  }
  else{
    workers.parallel_for(0, S, [&](size_t begin, size_t end){
      for(size_t i = begin; i < end; i++){
        float sum = 0.0;
        for(size_t j = 0; j < S; j++) sum += gradient[j] * activation->fun_prime(layer, j, i);
        output[i] = sum;
      };
    });
    workers.wait();
  };

  std::memcpy(sigma, output, sizeof(output));
  cpu_sigma_dirty = false;
  gpu_sigma_dirty = true;

  workers.parallel_for(0, D, [&](size_t begin, size_t end){
    for(size_t j = begin; j < end; j++){
      const float factor = learning_rate * sigma_next[j];
      float* row = weights + j * (S + 1);
      for(size_t i = 0; i < S; i++) row[i] -= factor * activated[i];
      row[S] -= factor;
    };
  });

  workers.wait();

  gpu_weights_dirty = true;
  cpu_weights_dirty = false;
  glFinish();
};



template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::backprop_gpu(Layer<D, N>& next_layer) noexcept {
  glFinish();
  unsigned int activation_id = 0;

  if(dynamic_cast<NN::Linear*>(activation.get())) activation_id = 1;
  else if(dynamic_cast<NN::Sigmoid*>(activation.get())) activation_id = 2;
  else if(dynamic_cast<NN::Softmax*>(activation.get())) activation_id = 3;
  else if(dynamic_cast<NN::ReLU*>(activation.get())) activation_id = 4;

  if(activation_id == 0) return;

  auto& gpu = NN::GPUAcceleration::get();

  ensure_gpu_storage();
  next_layer.ensure_gpu_storage();

  next_layer.gpu_storage[GPUSigma].bind(GPUBufferCount);

  auto& shader = gpu.getBackpropSigmaShader();
  
  glFinish();
  auto start = std::chrono::steady_clock::now();
  shader.run(gpu_storage, (S + 31) / 32);

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glFinish();
  for(GLenum error; (error = glGetError()) != GL_NO_ERROR;){
    printf("Backprop Sigma OpenGL error: 0x%X\n", error);
  }
  if(debug_logs){
    auto end = std::chrono::steady_clock::now();
    printf("backprop_gpu Sigma S=%u D=%u groups=%u: %.3f ms\n", S, D, (S + 31) / 32, std::chrono::duration<double, std::milli>(end - start).count());
  };

  if(activation_id == 3){
    auto& softmax_shader = gpu.getBackpropSoftmaxSigmaShader();
    glFinish();
    start = std::chrono::steady_clock::now();
    softmax_shader.run(gpu_storage, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();
    for(GLenum error; (error = glGetError()) != GL_NO_ERROR;){
      printf("Backprop Softmax OpenGL error: 0x%X\n", error);
    }
    if(debug_logs){
      auto end = std::chrono::steady_clock::now();
      printf("backprop_gpu softmax S=%u D=%u groups=%u: %.3f ms\n", S, D, 1u, std::chrono::duration<double, std::milli>(end - start).count());
    };
  };

  cpu_sigma_dirty = true;
  gpu_sigma_dirty = false;

  if(learning_rate != 0.0f){
    auto& update_shader = gpu.getBackpropUpdateShader();
    
    glFinish();
    start = std::chrono::steady_clock::now();
    update_shader.run(gpu_storage, (S + 256) / 256, D);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();
    for(GLenum error; (error = glGetError()) != GL_NO_ERROR;){
      printf("Backprop Update OpenGL error: 0x%X\n", error);
    }
    if(debug_logs){
      auto end = std::chrono::steady_clock::now();
      printf("backprop_gpu Update S=%u D=%u groups=(%u,%u): %.3f ms\n", S, D, (S + 256) / 256, D, std::chrono::duration<double, std::milli>(end - start).count());
    };

    glFinish();

    cpu_weights_dirty = true;
    gpu_weights_dirty = false;
  };
};



template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::backprop(Layer<D, N>& next_layer) noexcept {
  glFinish();
  activateNodes();
  const size_t work = size_t(S) * D;
  if(!gpu_acceleration){
    if(work > multithreading_acceleration_size_threshold) backprop_threads(next_layer);
    else backprop_cpu(next_layer);
  }
  else{
    backprop_gpu(next_layer);
  };
  activated_after_forward = false;
  glFinish();
};


// =========================== //
// ======= Presentation ====== //
// =========================== //
template <unsigned int S, unsigned int D>
inline std::string NN::Layer<S, D>::print() const {
  std::string s = "size: \n";

  s += std::to_string(S) + ", ";
  s += std::to_string(D) + "; ";

  // s += "\nloss, activation \n";
  // s += std::to_string(loss->getType()) + "; ";
  // if(activation)
    // s += std::to_string(activation->getType()) + "; ";
  // else
    // s += "-1; ";


  bool first = true;
  s += "\nnodes: \n[";
  for(float el : nodes){
    if(!first) 
      s += ", ";
      s += std::to_string(el);
      first = false;
  }
  s += "]; ";
  
  first = true;
  unsigned int i = 0;
  s += "\nweights: \n[[";
  for(float el : weights){
    if(!first && i % (S + 1) != 0)
      s += ", ";

    if(i % (S + 1) == 0 && i <= (S + 1) * D && i != 0) 
      s += "], \n[";

    s += std::to_string(el);
    first = false;
    i++;
  };
  s += "]];\n ";
  return s;
};



template <unsigned int S, unsigned int D>
inline std::string NN::Layer<S, D>::serialize() const noexcept {
  glFinish();
  std::string data;

  auto write = [&](Field id, const void* source, uint64_t size){
    uint32_t field = static_cast<uint32_t>(id);

    data.append(reinterpret_cast<const char*>(&field), sizeof(field));
    data.append(reinterpret_cast<const char*>(&size), sizeof(size));

    if(size > 0) data.append(reinterpret_cast<const char*>(source), size);
  };

  uint32_t s = S;
  uint32_t d = D;

  write(Field::S_val, &s, sizeof(s));
  write(Field::D_val, &d, sizeof(d));
  write(Field::LEARNING_RATE, &learning_rate, sizeof(learning_rate));

  uint32_t activation_id = 0;

  if(dynamic_cast<NN::Linear*>(activation.get()))
    activation_id = static_cast<uint32_t>(Activation::LINEAR);
  else if(dynamic_cast<NN::Sigmoid*>(activation.get()))
    activation_id = static_cast<uint32_t>(Activation::SIGMOID);
  else if(dynamic_cast<NN::Softmax*>(activation.get()))
    activation_id = static_cast<uint32_t>(Activation::SOFTMAX);
  else if(dynamic_cast<NN::ReLU*>(activation.get()))
    activation_id = static_cast<uint32_t>(Activation::RELU);

  write(Field::ACTIVATION, &activation_id, sizeof(activation_id));

  uint32_t loss_id = 0;

  if(dynamic_cast<NN::MSE*>(loss.get()))
    loss_id = static_cast<uint32_t>(Loss::MSE);
  else if(dynamic_cast<NN::CrossEntropy*>(loss.get()))
    loss_id = static_cast<uint32_t>(Loss::CROSSENTROPY);

  write(Field::LOSS, &loss_id, sizeof(loss_id));

  const float* current_weights = const_cast<NN::Layer<S, D>*>(this)->getWeights();

  write(Field::WEIGHTS, current_weights, sizeof(weights));

  uint32_t end = static_cast<uint32_t>(Field::END);

  data.append(reinterpret_cast<const char*>(&end), sizeof(end));
  glFinish();

  return data;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::deserialize(const std::string &data){
  glFinish();
  size_t offset = 0;

  auto read = [&](void* destination, size_t size){
    if(size > data.size() - offset) throw std::runtime_error("Invalid serialized layer data");

    std::memcpy(destination, data.data() + offset, size);
    offset += size;
  };

  while(offset < data.size()){
    uint32_t field_id = 0;

    read(&field_id, sizeof(field_id));

    Field field = static_cast<Field>(field_id);

    if(field == Field::END) break;

    uint64_t size = 0;

    read(&size, sizeof(size));

    if(size > data.size() - offset) throw std::runtime_error("Invalid serialized layer field size");

    switch(field){
      case Field::S_val:{
        if(size != sizeof(uint32_t))
          throw std::runtime_error("Invalid layer S size");

        uint32_t value = 0;

        read(&value, sizeof(value));

        if(value != S)
          throw std::runtime_error("Layer S size mismatch");

        break;
      };

      case Field::D_val:{
        if(size != sizeof(uint32_t))
          throw std::runtime_error("Invalid layer D size");

        uint32_t value = 0;

        read(&value, sizeof(value));

        if(value != D)
          throw std::runtime_error("Layer D size mismatch");

        break;
      };

      case Field::LEARNING_RATE:{
        if(size != sizeof(learning_rate))
          throw std::runtime_error("Invalid learning rate size");

        float value = 0.0f;

        read(&value, sizeof(value));
        setLearningRate(value);

        break;
      };

      case Field::ACTIVATION:{
        if(size != sizeof(uint32_t))
          throw std::runtime_error("Invalid activation size");

        uint32_t activation_id = 0;

        read(&activation_id, sizeof(activation_id));

        switch(static_cast<Activation>(activation_id)){
          case Activation::LINEAR:
            setActivation<NN::Linear>();
            break;

          case Activation::SIGMOID:
            setActivation<NN::Sigmoid>();
            break;

          case Activation::SOFTMAX:
            setActivation<NN::Softmax>();
            break;

          case Activation::RELU:
            setActivation<NN::ReLU>();
            break;

          default:
            throw std::runtime_error("Unknown activation type");
        };

        break;
      };

      case Field::LOSS:{
        if(size != sizeof(uint32_t))
          throw std::runtime_error("Invalid loss size");

        uint32_t loss_id = 0;

        read(&loss_id, sizeof(loss_id));

        switch(static_cast<Loss>(loss_id)){
          case Loss::MSE:
            setLoss<NN::MSE>();
            break;
          case Loss::CROSSENTROPY:
            setLoss<NN::CrossEntropy>();
            break;
          default:
            throw std::runtime_error("Unknown loss type");
        };
        break;
      };
      case Field::WEIGHTS:{
        if(size != sizeof(weights))
          throw std::runtime_error("Invalid weights size");
        read(weights, sizeof(weights));
        cpu_weights_dirty = false;
        gpu_weights_dirty = true;
        activated_after_forward = false;
        break;
      };
      default:{
        offset += static_cast<size_t>(size);
        break;
      };
    };
  };

  glFinish();
  gpu_header_dirty = true;
};