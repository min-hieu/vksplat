#ifndef VKGS_VIEWER_GUI_IMGUI_SOFTWARE_RENDERER_H
#define VKGS_VIEWER_GUI_IMGUI_SOFTWARE_RENDERER_H

#include <cstdint>

struct ImDrawData;

namespace vkgs {
namespace viewer {
namespace gui {

// Optimized software renderer for ImGui draw data
// Renders ImGui triangles to a BGRA pixel buffer with alpha blending
void RenderImGuiToBuffer(ImDrawData* draw_data, uint8_t* pixels, int width, int height, int pitch);

}  // namespace gui
}  // namespace viewer
}  // namespace vkgs

#endif  // VKGS_VIEWER_GUI_IMGUI_SOFTWARE_RENDERER_H

