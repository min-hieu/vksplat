#ifndef VKGS_CORE_DRAW_OPTIONS_H
#define VKGS_CORE_DRAW_OPTIONS_H

#include <glm/glm.hpp>

namespace vkgs {
namespace core {

struct DrawOptions {
  glm::mat4 view;
  glm::mat4 projection;
  uint32_t width;
  uint32_t height;
  glm::vec3 background;
  float eps2d;
  int sh_degree;
  bool visualize_depth = false;
  bool depth_auto_range = false;
  float depth_z_min = 22.0f;
  float depth_z_max = 50.0f;
  float camera_near = 0.1f;
  float camera_far = 1000.0f;
  // Output parameters for auto-range (filled in by Draw when depth_auto_range is true)
  float* depth_z_min_out = nullptr;
  float* depth_z_max_out = nullptr;
};

}  // namespace core
}  // namespace vkgs

#endif  // VKGS_CORE_DRAW_OPTIONS_H
