cmake_minimum_required(VERSION 3.10)

project(Tests LANGUAGES CXX)

set(src 
  test.cpp
)

add_executable(Tests ${src})

target_link_libraries(Tests DSL)