// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "GPUAcceleration.h"



// =========================== //
// ======= Constructors ====== //
// =========================== //
inline NN::GPUAcceleration::GPUAcceleration() noexcept {
  printf("OpenGL Vendor: %s\n", reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
  printf("OpenGL Renderer: %s\n", reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
  printf("OpenGL Version: %s\n", reinterpret_cast<const char*>(glGetString(GL_VERSION)));

  activate_Linear_shader.compile();
  activate_LinearPrime_shader.compile();

  activate_Sigmoid_shader.compile();
  activate_SigmoidPrime_shader.compile();

  activate_ReLU_shader.compile();
  activate_ReLUPrime_shader.compile();

  activate_Softmax_shader.compile();
  activate_SoftmaxPrime_shader.compile();

  loss_MSE_shader.compile();
  loss_MSEPrime_shader.compile();

  loss_CrossEntropy_shader.compile();
  loss_CrossEntropyPrime_shader.compile();

  backprop_SoftmaxCrossEntropy_shader.compile();

  forward_shader.compile();

  backprop_Sigma_shader.compile();
  backprop_SoftmaxSigma_shader.compile();
  backprop_Update_shader.compile();
  backprop_Initial_shader.compile();
};



inline NN::GPUAcceleration& NN::GPUAcceleration::get() noexcept {
  static GPUAcceleration instance;
  return instance;
};



// ============================== //
// ======= Compute Shaders ====== //
// ============================== //
inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getActivationLinearShader() noexcept {
  return activate_Linear_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getActivationLinearPrimeShader() noexcept {
  return activate_LinearPrime_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getActivationSigmoidShader() noexcept {
  return activate_Sigmoid_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getActivationSigmoidPrimeShader() noexcept {
  return activate_SigmoidPrime_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getActivationReLUShader() noexcept {
  return activate_ReLU_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getActivationReLUPrimeShader() noexcept {
  return activate_ReLUPrime_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getActivationSoftmaxShader() noexcept {
  return activate_Softmax_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getActivationSoftmaxPrimeShader() noexcept {
  return activate_SoftmaxPrime_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getForwardShader() noexcept {
  return forward_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getLossMSEShader() noexcept {
  return loss_MSE_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getLossMSEPrimeShader() noexcept {
  return loss_MSEPrime_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getLossCrossEntropyShader() noexcept {
  return loss_CrossEntropy_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getLossCrossEntropyPrimeShader() noexcept {
  return loss_CrossEntropyPrime_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getBackpropSoftmaxCrossEntropyShader() noexcept {
  return backprop_SoftmaxCrossEntropy_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getBackpropSigmaShader() noexcept {
  return backprop_Sigma_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getBackpropSoftmaxSigmaShader() noexcept {
  return backprop_SoftmaxSigma_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getBackpropInitialShader() noexcept {
  return backprop_Initial_shader;
};

inline CW::Renderer::ComputeShader& NN::GPUAcceleration::getBackpropUpdateShader() noexcept {
  return backprop_Update_shader;
};
