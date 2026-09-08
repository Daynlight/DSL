// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <string>



// ====================== //
// ======= Linear ======= //
// ====================== //

inline const std::string activate_Linear_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 2) readonly buffer Nodes {
  float nodes[];
};

layout(std430, binding = 3) writeonly buffer Activated {
  float activated[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= activated.length()) return;
  activated[i] = nodes[i];
}

)";

inline const std::string activate_LinearPrime_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 5) writeonly buffer Gradient {
  float gradient[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= gradient.length()) return;
  gradient[i] = 1.0;
}

)";



// ====================== //
// ======= Sigmoid ====== //
// ====================== //

inline const std::string activate_Sigmoid_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 2) readonly buffer Nodes {
  float nodes[];
};

layout(std430, binding = 3) writeonly buffer Activated {
  float activated[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= activated.length()) return;

  float x = nodes[i];

  if(x >= 0.0){
    activated[i] = 1.0 / (1.0 + exp(-x));
  }
  else{
    float z = exp(x);
    activated[i] = z / (1.0 + z);
  }
}

)";

inline const std::string activate_SigmoidPrime_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 5) writeonly buffer Gradient {
  float gradient[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= activated.length()) return;

  float y = activated[i];
  gradient[i] = y * (1.0 - y);
}

)";



// ====================== //
// ======== ReLU ======== //
// ====================== //

inline const std::string activate_ReLU_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 2) readonly buffer Nodes {
  float nodes[];
};

layout(std430, binding = 3) writeonly buffer Activated {
  float activated[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= activated.length()) return;
  activated[i] = max(nodes[i], 0.0);
}

)";

inline const std::string activate_ReLUPrime_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 2) readonly buffer Nodes {
  float nodes[];
};

layout(std430, binding = 5) writeonly buffer Gradient {
  float gradient[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= gradient.length()) return;
  gradient[i] = nodes[i] > 0.0 ? 1.0 : 0.0;
}

)";



// ====================== //
// ======= Softmax ====== //
// ====================== //

inline const std::string activate_Softmax_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 2) readonly buffer Nodes {
  float nodes[];
};

layout(std430, binding = 3) writeonly buffer Activated {
  float activated[];
};

shared float partial[256];

void main(){
  uint lane = gl_LocalInvocationID.x;
  uint size = uint(activated.length());

  if(size == 0u) return;

  float localMax = -3.402823466e38;

  for(uint i = lane; i < size; i += 256u)
    localMax = max(localMax, nodes[i]);

  partial[lane] = localMax;
  barrier();

  for(uint stride = 128u; stride > 0u; stride >>= 1u){
    if(lane < stride) partial[lane] = max(partial[lane], partial[lane + stride]);
    barrier();
  }

  float maxValue = partial[0];
  float localSum = 0.0;

  for(uint i = lane; i < size; i += 256u)
    localSum += exp(nodes[i] - maxValue);

  partial[lane] = localSum;
  barrier();

  for(uint stride = 128u; stride > 0u; stride >>= 1u){
    if(lane < stride) partial[lane] += partial[lane + stride];
    barrier();
  }

  float sum = partial[0];

  for(uint i = lane; i < size; i += 256u)
    activated[i] = exp(nodes[i] - maxValue) / sum;
}

)";

inline const std::string activate_SoftmaxPrime_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 5) readonly buffer Gradient {
  float gradient[];
};

layout(std430, binding = 4) writeonly buffer Sigma {
  float sigma[];
};

shared float partial[256];

void main(){
  uint lane = gl_LocalInvocationID.x;
  uint size = uint(activated.length());

  if(size == 0u) return;

  float dot = 0.0;

  for(uint i = lane; i < size; i += 256u)
    dot += activated[i] * gradient[i];

  partial[lane] = dot;
  barrier();

  for(uint stride = 128u; stride > 0u; stride >>= 1u){
    if(lane < stride) partial[lane] += partial[lane + stride];
    barrier();
  }

  dot = partial[0];

  for(uint i = lane; i < size; i += 256u)
    sigma[i] = activated[i] * (gradient[i] - dot);
}

)";



// ====================== //
// ========= MSE ======== //
// ====================== //

inline const std::string loss_MSE_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 6) readonly buffer Target {
  float target[];
};

layout(std430, binding = 5) writeonly buffer Gradient {
  float gradient[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= activated.length()) return;

  float x = activated[i];
  float t = target[i];

  gradient[i] = (x - t) * (x - t) / 2.0;
}

)";

inline const std::string loss_MSEPrime_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 6) readonly buffer Target {
  float target[];
};

layout(std430, binding = 5) writeonly buffer Gradient {
  float gradient[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= activated.length()) return;
  gradient[i] = activated[i] - target[i];
}

)";



// ====================== //
// ==== CrossEntropy ==== //
// ====================== //

inline const std::string loss_CrossEntropy_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 6) readonly buffer Target {
  float target[];
};

layout(std430, binding = 5) writeonly buffer Gradient {
  float gradient[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= activated.length()) return;

  float x = clamp(activated[i], 1e-7, 1.0);
  float t = target[i];

  gradient[i] = -t * log(x);
}

)";

inline const std::string loss_CrossEntropyPrime_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 6) readonly buffer Target {
  float target[];
};

layout(std430, binding = 5) writeonly buffer Gradient {
  float gradient[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= activated.length()) return;

  float t = target[i];

  if(t == 0.0){
    gradient[i] = 0.0;
    return;
  }

  float x = clamp(activated[i], 1e-7, 1.0);
  gradient[i] = -t / x;
}

)";



// ====================== //
// ======= Forward ====== //
// ====================== //

inline const std::string forward_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Header {
  float meta[];
};

layout(std430, binding = 1) readonly buffer Weights {
  float weights[];
};

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 7) writeonly buffer NextNodes {
  float next_nodes[];
};

shared float partial[256];

void main(){
  uint i = gl_WorkGroupID.x;
  uint lane = gl_LocalInvocationID.x;

  uint S = uint(meta[0]);
  uint D = uint(meta[1]);

  if(i >= D) return;

  uint row = i * (S + 1u);
  float sum = 0.0;

  for(uint j = lane; j < S; j += 256u)
    sum += activated[j] * weights[row + j];

  partial[lane] = sum;
  barrier();

  for(uint stride = 128u; stride > 0u; stride >>= 1u){
    if(lane < stride)
      partial[lane] += partial[lane + stride];

    barrier();
  }

  if(lane == 0u)
    next_nodes[i] = partial[0] + weights[row + S];
}

)";



// ====================== //
// ====== Backprop ====== //
// ====================== //

inline const std::string backprop_SoftmaxCrossEntropy_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 4) writeonly buffer Sigma {
  float sigma[];
};

layout(std430, binding = 6) readonly buffer Target {
  float target[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= sigma.length()) return;
  sigma[i] = activated[i] - target[i];
}

)";

inline const std::string backprop_Initial_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Header {
  float meta[];
};

layout(std430, binding = 2) readonly buffer Nodes {
  float nodes[];
};

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 4) writeonly buffer Sigma {
  float sigma[];
};

layout(std430, binding = 6) readonly buffer Target {
  float target[];
};

shared float partial[256];

float loss_gradient(uint i, uint loss_id){
  float p = activated[i];
  float t = target[i];

  if(loss_id == 1u) return p - t;
  if(t == 0.0) return 0.0;

  return -t / clamp(p, 1e-7, 1.0);
}

void main(){
  uint i = gl_GlobalInvocationID.x;
  uint S = uint(meta[0]);
  uint activation_id = uint(meta[2]);
  uint loss_id = uint(meta[3]);

  if(activation_id == 3u){
    uint lane = gl_LocalInvocationID.x;
    float dot = 0.0;

    for(uint j = lane; j < S; j += 256u)
      dot += activated[j] * loss_gradient(j, loss_id);

    partial[lane] = dot;
    barrier();

    for(uint stride = 128u; stride > 0u; stride >>= 1u){
      if(lane < stride) partial[lane] += partial[lane + stride];
      barrier();
    }

    dot = partial[0];

    for(uint j = lane; j < S; j += 256u)
      sigma[j] = activated[j] * (loss_gradient(j, loss_id) - dot);

    return;
  }

  if(i >= S) return;

  float gradient = loss_gradient(i, loss_id);

  if(activation_id == 1u) sigma[i] = gradient;
  else if(activation_id == 2u) sigma[i] = gradient * activated[i] * (1.0 - activated[i]);
  else if(activation_id == 4u) sigma[i] = gradient * (nodes[i] > 0.0 ? 1.0 : 0.0);
}

)";

inline const std::string backprop_Sigma_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Header {
  float meta[];
};

layout(std430, binding = 1) readonly buffer Weights {
  float weights[];
};

layout(std430, binding = 2) readonly buffer Nodes {
  float nodes[];
};

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 4) writeonly buffer Sigma {
  float sigma[];
};

layout(std430, binding = 5) writeonly buffer Gradient {
  float gradient[];
};

layout(std430, binding = 7) readonly buffer NextSigma {
  float next_sigma[];
};

void main(){
  uint i = gl_GlobalInvocationID.x;

  uint S = uint(meta[0]);
  uint D = uint(meta[1]);
  uint activation_id = uint(meta[2]);

  if(i >= S) return;

  float value = 0.0;

  for(uint j = 0; j < D; j++)
    value += weights[j * (S + 1u) + i] * next_sigma[j];

  gradient[i] = value;

  if(activation_id == 1u) sigma[i] = value;
  else if(activation_id == 2u) sigma[i] = value * activated[i] * (1.0 - activated[i]);
  else if(activation_id == 4u) sigma[i] = value * (nodes[i] > 0.0 ? 1.0 : 0.0);
}

)";

inline const std::string backprop_SoftmaxSigma_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Header {
  float meta[];
};

layout(std430, binding = 1) readonly buffer Weights {
  float weights[];
};

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 4) writeonly buffer Sigma {
  float sigma[];
};

layout(std430, binding = 5) buffer Gradient {
  float gradient[];
};

layout(std430, binding = 7) readonly buffer NextSigma {
  float next_sigma[];
};

shared float partial[256];

float calculate_gradient(uint i, uint S, uint D){
  float sum = 0.0;

  for(uint j = 0; j < D; j++)
    sum += weights[j * (S + 1u) + i] * next_sigma[j];

  return sum;
}

void main(){
  uint lane = gl_LocalInvocationID.x;
  uint S = uint(meta[0]);
  uint D = uint(meta[1]);

  float dot = 0.0;

  for(uint i = lane; i < S; i += 256u){
    float value = calculate_gradient(i, S, D);
    gradient[i] = value;
    dot += activated[i] * value;
  }

  partial[lane] = dot;
  barrier();

  for(uint stride = 128u; stride > 0u; stride >>= 1u){
    if(lane < stride) partial[lane] += partial[lane + stride];
    barrier();
  }

  dot = partial[0];

  for(uint i = lane; i < S; i += 256u)
    sigma[i] = activated[i] * (gradient[i] - dot);
}

)";

inline const std::string backprop_Update_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Header {
  float meta[];
};

layout(std430, binding = 1) buffer Weights {
  float weights[];
};

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 7) readonly buffer NextSigma {
  float next_sigma[];
};

void main(){
  uint j = gl_GlobalInvocationID.x;

  uint S = uint(meta[0]);
  uint D = uint(meta[1]);
  float learning_rate = meta[4];

  if(j >= D) return;

  uint row = j * (S + 1u);
  float factor = learning_rate * next_sigma[j];

  for(uint i = 0; i < S; i++)
    weights[row + i] -= factor * activated[i];

  weights[row + S] -= factor;
}

)";