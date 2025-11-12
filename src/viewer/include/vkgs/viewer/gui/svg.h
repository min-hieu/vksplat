#ifndef VKGS_VIEWER_GUI_SVG_H
#define VKGS_VIEWER_GUI_SVG_H

#include <string>
#include <vector>
#include <cstdint>

namespace vkgs {
namespace viewer {
namespace gui {

class SVG {
 public:
  SVG();
  ~SVG();

  void RenderImage(std::vector<uint8_t>& image_data, int width, int height,
                   const std::string& svg_path, int x, int y, int svg_width, int svg_height,
                   uint8_t tint_r = 255, uint8_t tint_g = 255, uint8_t tint_b = 255) const;
};

}  // namespace gui
}  // namespace viewer
}  // namespace vkgs

#endif  // VKGS_VIEWER_GUI_SVG_H

