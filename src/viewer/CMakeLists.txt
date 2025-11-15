cmake_minimum_required(VERSION 3.15)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Add ImGui
include(FetchContent)
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        docking
)
FetchContent_MakeAvailable(imgui)

# ImGui is header-only, but we need to add the source files for the implementation
set(IMGUI_SOURCES
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

# Find SDL3
find_package(SDL3 QUIET)
if(NOT SDL3_FOUND)
  # Try to find SDL3 manually
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
  add_library(vkgs_viewer STATIC
    src/swapchain.cc
    src/viewer.cc
    src/gui.cc
    src/gui/font.cc
    src/gui/svg.cc
    src/gui/title_screen.cc
    src/gui/stats_panel.cc
    src/gui/visual_panel.cc
    ${IMGUI_SOURCES}
  )

  # On macOS, compile viewer.cc as Objective-C++ for file picker
  if(APPLE)
    set_source_files_properties(src/viewer.cc PROPERTIES LANGUAGE OBJCXX)
  endif()

  target_include_directories(vkgs_viewer PUBLIC
    include
    ${SDL3_INCLUDE_DIR}
  )

  target_include_directories(vkgs_viewer PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    /usr/local/include  # Fallback for Vulkan headers
    ${CMAKE_SOURCE_DIR}/third_party
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
  )

  target_link_libraries(vkgs_viewer
    PRIVATE
      vkgs::core
      vkgs::gpu
      ${SDL3_LIBRARY}
  )

  if(SDL3_FOUND)
    target_link_libraries(vkgs_viewer PRIVATE SDL3::SDL3)
  endif()

  # On macOS, link with Cocoa, IOKit, and CoreVideo frameworks
  if(APPLE)
    target_link_libraries(vkgs_viewer PRIVATE
      "-framework Cocoa"
      "-framework IOKit"
      "-framework CoreVideo"
    )
  endif()

  target_compile_features(vkgs_viewer PUBLIC cxx_std_17)
endif()

