cmake_minimum_required(VERSION 3.15)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find SDL3
find_package(SDL3 QUIET)
if(NOT SDL3_FOUND)
  find_path(SDL3_INCLUDE_DIR SDL3/SDL.h
    PATHS
      /opt/homebrew/include
      /usr/local/include
      /usr/include
  )

  find_library(SDL3_LIBRARY
    NAMES SDL3
    PATHS
      /opt/homebrew/lib
      /usr/local/lib
      /usr/lib
  )

  if(SDL3_INCLUDE_DIR AND SDL3_LIBRARY)
    set(SDL3_FOUND TRUE)
  endif()
endif()

if(SDL3_FOUND)
  add_executable(inspector
    inspector.cc
  )

  target_include_directories(inspector PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    /usr/local/include  # Fallback for Vulkan headers
    ${CMAKE_BINARY_DIR}/_deps/imgui-src
    ${CMAKE_BINARY_DIR}/_deps/imgui-src/backends
  )

  target_link_libraries(inspector
    PRIVATE
      vkgs::core
      vkgs::gpu
      vkgs_viewer
  )

  set_target_properties(inspector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
  )
endif()

