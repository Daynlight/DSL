cmake_minimum_required(VERSION 3.10)

project(DSL LANGUAGES CXX)

set(headers
  Essentials/
  "Math/"
)

set(src
  dsl.cpp
  Essentials/Vector/vector.hpp
  Math/ModuloZ/ModuloZ.hpp
)

add_library(DSL STATIC ${headers} ${src})

target_include_directories(DSL PUBLIC ${headers})