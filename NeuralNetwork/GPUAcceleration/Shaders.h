// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <string>



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
  precise float sum = 0.0;

  for(uint j = lane; j < S; j += 256u){
    precise float product = activated[j] * weights[row + j];
    sum = sum + product;
  }

  partial[lane] = sum;
  barrier();

  for(uint stride = 128u; stride > 0u; stride >>= 1u){
    if(lane < stride)
      partial[lane] = partial[lane] + partial[lane + stride];

    barrier();
  }

  if(lane == 0u)
    next_nodes[i] = partial[0] + weights[row + S];
}

)";





// ====================== //
// ===== Hidden Sigma === //
// ====================== //

inline const std::string backprop_Sigma_shader_src = R"(

#version 430 core

layout(local_size_x = 32, local_size_y = 8) in;

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

shared float partial[8][32];

void main(){
  uint lane_x = gl_LocalInvocationID.x;
  uint lane_y = gl_LocalInvocationID.y;

  uint S = uint(meta[0]);
  uint D = uint(meta[1]);
  uint activation_id = uint(meta[2]);

  uint i = gl_WorkGroupID.x * 32u + lane_x;
  uint row_size = S + 1u;

  precise float value = 0.0;

  if(i < S){
    for(uint j = lane_y; j < D; j += 8u){
      precise float product = weights[j * row_size + i] * next_sigma[j];
      value = value + product;
    }
  }

  partial[lane_y][lane_x] = value;
  barrier();

  for(uint stride = 4u; stride > 0u; stride >>= 1u){
    if(lane_y < stride)
      partial[lane_y][lane_x] = partial[lane_y][lane_x] + partial[lane_y + stride][lane_x];

    barrier();
  }

  if(lane_y == 0u && i < S){
    value = partial[0][lane_x];
    gradient[i] = value;

    if(activation_id == 1u) sigma[i] = value;
    else if(activation_id == 2u){
      float a = activated[i];
      sigma[i] = value * a * (1.0 - a);
    }
    else if(activation_id == 4u)
      sigma[i] = nodes[i] > 0.0 ? value : 0.0;
  }
}

)";





// ====================== //
// === Softmax Sigma ==== //
// ====================== //

inline const std::string backprop_SoftmaxSigma_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Header {
  float meta[];
};

layout(std430, binding = 3) readonly buffer Activated {
  float activated[];
};

layout(std430, binding = 4) writeonly buffer Sigma {
  float sigma[];
};

layout(std430, binding = 5) readonly buffer Gradient {
  float gradient[];
};

shared float partial[256];

void main(){
  uint lane = gl_LocalInvocationID.x;
  uint S = uint(meta[0]);

  precise float dot = 0.0;

  for(uint i = lane; i < S; i += 256u){
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

  for(uint i = lane; i < S; i += 256u)
    sigma[i] = activated[i] * (gradient[i] - dot);
}

)";





// ====================== //
// === Update Weights === //
// ====================== //

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
  uint S = uint(meta[0]);
  uint D = uint(meta[1]);

  uint row = gl_WorkGroupID.y;
  uint col = gl_WorkGroupID.x * 256u + gl_LocalInvocationID.x;

  if(row >= D || col > S) return;

  uint index = row * (S + 1u) + col;

  precise float factor = meta[4] * next_sigma[row];
  float input_value = col == S ? 1.0 : activated[col];
  precise float delta = factor * input_value;

  weights[index] = weights[index] - delta;
}

)";





// ====================== //
// ====== Backprop ====== //
// ====================== //

inline const std::string backprop_SoftmaxCrossEntropy_shader_src = R"(

#version 430 core

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Header {
  float meta[];
};

layout(std430, binding = 2) readonly buffer Nodes {
  float nodes[];
};

layout(std430, binding = 4) writeonly buffer Sigma {
  float sigma[];
};

layout(std430, binding = 6) readonly buffer Target {
  float target[];
};

shared float partial_max[256];
shared float partial_sum[256];

void main(){
  uint lane = gl_LocalInvocationID.x;
  uint S = uint(meta[0]);

  float maximum = -3.402823466e+38;

  for(uint i = lane; i < S; i += 256u)
    maximum = max(maximum, nodes[i]);

  partial_max[lane] = maximum;
  barrier();

  for(uint stride = 128u; stride > 0u; stride >>= 1u){
    if(lane < stride)
      partial_max[lane] = max(partial_max[lane], partial_max[lane + stride]);

    barrier();
  }

  maximum = partial_max[0];

  precise float sum = 0.0;

  for(uint i = lane; i < S; i += 256u)
    sum = sum + exp(nodes[i] - maximum);

  partial_sum[lane] = sum;
  barrier();

  for(uint stride = 128u; stride > 0u; stride >>= 1u){
    if(lane < stride)
      partial_sum[lane] = partial_sum[lane] + partial_sum[lane + stride];

    barrier();
  }

  sum = partial_sum[0];

  for(uint i = lane; i < S; i += 256u){
    float probability = exp(nodes[i] - maximum) / sum;
    sigma[i] = probability - target[i];
  }
}

)";





// ====================== //
// === Initial Sigma ==== //
// ====================== //

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
shared float partial_max[256];
shared float partial_sum[256];

float loss_gradient(uint i, uint loss_id){
  float p = activated[i];
  float t = target[i];

  if(loss_id == 1u) return p - t;
  if(t == 0.0) return 0.0;

  return -t / clamp(p, 1e-7, 1.0);
}

void main(){
  uint i = gl_GlobalInvocationID.x;
  uint lane = gl_LocalInvocationID.x;
  uint S = uint(meta[0]);
  uint activation_id = uint(meta[2]);
  uint loss_id = uint(meta[3]);

  if(activation_id == 3u && loss_id == 2u){
    float maximum = -3.402823466e+38;

    for(uint j = lane; j < S; j += 256u)
      maximum = max(maximum, nodes[j]);

    partial_max[lane] = maximum;
    barrier();

    for(uint stride = 128u; stride > 0u; stride >>= 1u){
      if(lane < stride)
        partial_max[lane] = max(partial_max[lane], partial_max[lane + stride]);

      barrier();
    }

    maximum = partial_max[0];

    precise float sum = 0.0;

    for(uint j = lane; j < S; j += 256u)
      sum = sum + exp(nodes[j] - maximum);

    partial_sum[lane] = sum;
    barrier();

    for(uint stride = 128u; stride > 0u; stride >>= 1u){
      if(lane < stride)
        partial_sum[lane] = partial_sum[lane] + partial_sum[lane + stride];

      barrier();
    }

    sum = partial_sum[0];

    for(uint j = lane; j < S; j += 256u){
      float probability = exp(nodes[j] - maximum) / sum;
      sigma[j] = probability - target[j];
    }

    return;
  }

  if(activation_id == 3u){
    precise float dot = 0.0;

    for(uint j = lane; j < S; j += 256u){
      precise float product = activated[j] * loss_gradient(j, loss_id);
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

    for(uint j = lane; j < S; j += 256u)
      sigma[j] = activated[j] * (loss_gradient(j, loss_id) - dot);

    return;
  }

  if(i >= S) return;

  float gradient = loss_gradient(i, loss_id);

  if(activation_id == 1u) sigma[i] = gradient;
  else if(activation_id == 2u){
    float a = activated[i];
    sigma[i] = gradient * a * (1.0 - a);
  }
  else if(activation_id == 4u)
    sigma[i] = nodes[i] > 0.0 ? gradient : 0.0;
}

)";