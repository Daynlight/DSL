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
    precise float z = exp(-x);
    activated[i] = 1.0 / (1.0 + z);
  }
  else{
    precise float z = exp(x);
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
  if(i >= gradient.length()) return;

  float y = activated[i];
  precise float derivative = y * (1.0 - y);

  gradient[i] = derivative;
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

shared precise float partial[256];

void main(){
  uint lane = gl_LocalInvocationID.x;
  uint size = uint(activated.length());

  if(size == 0u) return;

  float localMax = -3.402823466e+38;

  for(uint i = lane; i < size; i += 256u)
    localMax = max(localMax, nodes[i]);

  partial[lane] = localMax;
  barrier();

  for(uint stride = 128u; stride > 0u; stride >>= 1u){
    if(lane < stride)
      partial[lane] = max(partial[lane], partial[lane + stride]);

    barrier();
  }

  float maxValue = partial[0];
  precise float localSum = 0.0;

  for(uint i = lane; i < size; i += 256u){
    precise float value = exp(nodes[i] - maxValue);
    localSum = localSum + value;
  }

  partial[lane] = localSum;
  barrier();

  for(uint stride = 128u; stride > 0u; stride >>= 1u){
    if(lane < stride)
      partial[lane] = partial[lane] + partial[lane + stride];

    barrier();
  }

  precise float sum = partial[0];

  for(uint i = lane; i < size; i += 256u){
    precise float value = exp(nodes[i] - maxValue);
    activated[i] = value / sum;
  }
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

shared precise float partial[256];

void main(){
  uint lane = gl_LocalInvocationID.x;
  uint size = uint(activated.length());

  if(size == 0u) return;

  precise float dot = 0.0;

  for(uint i = lane; i < size; i += 256u){
    precise float product = activated[i] * gradient[i];
    dot = dot + product;
  }

  partial[lane] = dot;
  barrier();

  for(uint stride = 128u; stride > 0u; stride >>= 1u){
    if(lane < stride)
      partial[lane] = partial[lane] + partial[lane + stride];

    barrier();
  }

  dot = partial[0];

  for(uint i = lane; i < size; i += 256u){
    precise float difference = gradient[i] - dot;
    sigma[i] = activated[i] * difference;
  }
}

)";