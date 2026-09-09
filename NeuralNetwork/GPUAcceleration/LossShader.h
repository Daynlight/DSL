// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once
#include <string>



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
  if(i >= gradient.length()) return;

  precise float diff = activated[i] - target[i];
  gradient[i] = (diff * diff) * 0.5;
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
  if(i >= gradient.length()) return;

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
  if(i >= gradient.length()) return;

  float t = target[i];

  if(t == 0.0){
    gradient[i] = 0.0;
    return;
  }

  float x = clamp(activated[i], 1e-7, 1.0);
  precise float value = -t * log(x);

  gradient[i] = value;
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
  if(i >= gradient.length()) return;

  float t = target[i];

  if(t == 0.0){
    gradient[i] = 0.0;
    return;
  }

  float x = clamp(activated[i], 1e-7, 1.0);
  precise float value = -t / x;

  gradient[i] = value;
}

)";