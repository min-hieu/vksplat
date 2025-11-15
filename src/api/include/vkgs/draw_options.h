#ifndef VKGS_DRAW_OPTIONS_H
#define VKGS_DRAW_OPTIONS_H

#include <cstdint>

namespace vkgs {

struct DrawOptions {
  float view[16];        // column-major
  float projection[16];  // column-major
  uint32_t width;
  uint32_t height;
  float background[3];
  float eps2d;
  int sh_degree;
  bool visualize_depth = false;
  bool depth_auto_range = false;
  float depth_z_min = 22.0f;
  float depth_z_max = 50.0f;
  float camera_near = 0.1f;
  float camera_far = 1000.0f;
};

}  // namespace vkgs

#endif  // VKGS_DRAW_OPTIONS_H
