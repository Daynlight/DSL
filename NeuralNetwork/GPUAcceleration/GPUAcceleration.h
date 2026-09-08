// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <vector>
#include <string>
#include <CWindow/Renderer/OpenGL/Renderer.h>

#include "Shaders.h"
#include "ActivationShader.h"
#include "LossShader.h"


namespace NN {
class GPUAcceleration{
// ======================================== //
// ================= Data ================= //
// ======================================== //
private:
  CW::Renderer::Renderer renderer = CW::Renderer::Renderer(true);
  CW::Renderer::ComputeShader activate_Linear_shader = CW::Renderer::ComputeShader(activate_Linear_shader_src);
  CW::Renderer::ComputeShader activate_LinearPrime_shader = CW::Renderer::ComputeShader(activate_LinearPrime_shader_src);
  CW::Renderer::ComputeShader activate_Sigmoid_shader = CW::Renderer::ComputeShader(activate_Sigmoid_shader_src);
  CW::Renderer::ComputeShader activate_SigmoidPrime_shader = CW::Renderer::ComputeShader(activate_SigmoidPrime_shader_src);
  CW::Renderer::ComputeShader activate_ReLU_shader = CW::Renderer::ComputeShader(activate_ReLU_shader_src);
  CW::Renderer::ComputeShader activate_ReLUPrime_shader = CW::Renderer::ComputeShader(activate_ReLUPrime_shader_src);
  CW::Renderer::ComputeShader activate_Softmax_shader = CW::Renderer::ComputeShader(activate_Softmax_shader_src);
  CW::Renderer::ComputeShader activate_SoftmaxPrime_shader = CW::Renderer::ComputeShader(activate_SoftmaxPrime_shader_src);
  CW::Renderer::ComputeShader loss_MSE_shader = CW::Renderer::ComputeShader(loss_MSE_shader_src);
  CW::Renderer::ComputeShader loss_MSEPrime_shader = CW::Renderer::ComputeShader(loss_MSEPrime_shader_src);
  CW::Renderer::ComputeShader loss_CrossEntropy_shader = CW::Renderer::ComputeShader(loss_CrossEntropy_shader_src);
  CW::Renderer::ComputeShader loss_CrossEntropyPrime_shader = CW::Renderer::ComputeShader(loss_CrossEntropyPrime_shader_src);
  CW::Renderer::ComputeShader forward_shader = CW::Renderer::ComputeShader(forward_shader_src);
  CW::Renderer::ComputeShader backprop_SoftmaxCrossEntropy_shader = CW::Renderer::ComputeShader(backprop_SoftmaxCrossEntropy_shader_src);
  CW::Renderer::ComputeShader backprop_Sigma_shader = CW::Renderer::ComputeShader(backprop_Sigma_shader_src);
  CW::Renderer::ComputeShader backprop_SoftmaxSigma_shader = CW::Renderer::ComputeShader(backprop_SoftmaxSigma_shader_src);
  CW::Renderer::ComputeShader backprop_Update_shader = CW::Renderer::ComputeShader(backprop_Update_shader_src);
  CW::Renderer::ComputeShader backprop_Initial_shader = CW::Renderer::ComputeShader(backprop_Initial_shader_src);



// ======================================== //
// ============== Functions =============== //
// ======================================== //
// =========================== //
// ======= Constructors ====== //
// =========================== //
public:
  GPUAcceleration() noexcept;
  ~GPUAcceleration() noexcept = default;
  GPUAcceleration(const GPUAcceleration& second) noexcept = delete;
  GPUAcceleration& operator=(const GPUAcceleration& second) noexcept = delete;
  GPUAcceleration(GPUAcceleration&& second) noexcept = delete;
  GPUAcceleration& operator=(GPUAcceleration&& second) noexcept = delete;
  static GPUAcceleration& get() noexcept;

// ============================== //
// ======= Compute Shaders ====== //
// ============================== //
public:
  CW::Renderer::ComputeShader& getActivationLinearShader() noexcept;
  CW::Renderer::ComputeShader& getActivationLinearPrimeShader() noexcept;
  CW::Renderer::ComputeShader& getActivationSigmoidShader() noexcept;
  CW::Renderer::ComputeShader& getActivationSigmoidPrimeShader() noexcept;
  CW::Renderer::ComputeShader& getActivationReLUShader() noexcept;
  CW::Renderer::ComputeShader& getActivationReLUPrimeShader() noexcept;
  CW::Renderer::ComputeShader& getActivationSoftmaxShader() noexcept;
  CW::Renderer::ComputeShader& getActivationSoftmaxPrimeShader() noexcept;
  CW::Renderer::ComputeShader& getLossMSEShader() noexcept;
  CW::Renderer::ComputeShader& getLossMSEPrimeShader() noexcept;
  CW::Renderer::ComputeShader& getLossCrossEntropyShader() noexcept;
  CW::Renderer::ComputeShader& getLossCrossEntropyPrimeShader() noexcept;
  CW::Renderer::ComputeShader& getBackpropSoftmaxCrossEntropyShader() noexcept;
  CW::Renderer::ComputeShader& getBackpropSigmaShader() noexcept;
  CW::Renderer::ComputeShader& getBackpropSoftmaxSigmaShader() noexcept;
  CW::Renderer::ComputeShader& getBackpropUpdateShader() noexcept;
  CW::Renderer::ComputeShader& getBackpropInitialShader() noexcept;
  CW::Renderer::ComputeShader& getForwardShader() noexcept;
};
}



#include "GPUAcceleration.hpp"
