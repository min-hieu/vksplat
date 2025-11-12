cmake_minimum_required(VERSION 3.15)

project(vkgs LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/add_shader.cmake)

add_subdirectory(src/gpu)
add_subdirectory(src/core)
add_subdirectory(src/api)
add_subdirectory(src/viewer)
add_subdirectory(inspector)
