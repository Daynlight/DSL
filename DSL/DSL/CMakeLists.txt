cmake_minimum_required(VERSION 3.10)

project(DSL LANGUAGES CXX)

set(src
  dsl.cpp
)

set(header
  dsl.h
  Vector/vector.h
  Vector/vector.hpp
)


add_library(DSL STATIC ${header} ${src})