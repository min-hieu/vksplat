#include <iostream>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "vkgs/core/renderer.h"
#include "vkgs/viewer/viewer.h"

int main(int argc, char* argv[]) {
  std::string ply_path;
  if (argc >= 2) {
    ply_path = argv[1];
  }

  try {
    // Initialize SDL3 first so Device can get required extensions
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
      std::cerr << "Failed to initialize SDL3: " << SDL_GetError() << std::endl;
      return 1;
    }

    auto renderer = std::make_shared<vkgs::core::Renderer>();
    std::cout << "Device: " << renderer->device_name() << std::endl;

    vkgs::viewer::Viewer viewer(renderer, "Coolant Visual Inspector", 1280, 720);
    viewer.Run(ply_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

