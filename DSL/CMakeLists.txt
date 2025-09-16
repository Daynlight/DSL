cmake_minimum_required(VERSION 3.10)

add_subdirectory(DSL)

project(Main LANGUAGES CXX)

set(headers
  DSL
  vendor/
  Tests
  Examples
)

set(src 
  main.cpp
)

add_executable(Main ${src} ${headers})

target_include_directories(Main PRIVATE ${headers})
target_link_libraries(Main PRIVATE DSL)
target_link_libraries(Main PRIVATE fmt)
