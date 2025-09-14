cmake_minimum_required(VERSION 3.10)

project(DSL LANGUAGES C)

set(src
  Vector/vector.c
)

set(header
  dsl.h
  Vector/vector.h
)


add_library(DSL STATIC ${header} ${src})