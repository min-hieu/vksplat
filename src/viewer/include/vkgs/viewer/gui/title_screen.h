#ifndef VKGS_VIEWER_GUI_TITLE_SCREEN_H
#define VKGS_VIEWER_GUI_TITLE_SCREEN_H

#include <string>
#include <vector>
#include <cstdint>
#include <functional>

struct ImFont;
struct SDL_Window;
struct ImDrawData;

// Include ImGui for ImTextureID
#include "imgui.h"

namespace vkgs {
namespace viewer {
namespace gui {

class TitleScreen {
 public:
  TitleScreen(const std::string& assets_path);
  ~TitleScreen();

  void Initialize(SDL_Window* window);
  void RenderUI(std::string& pending_ply_path,
                std::function<std::string()> show_file_picker,
                std::function<ImTextureID(const std::string&, int, int)> load_svg_texture);
  bool HandleClick(int x, int y, int width, int height,
                   std::string& pending_ply_path,
                   std::function<std::string()> show_file_picker);

  std::string assets_path_;
  std::string font_path_;
  SDL_Window* window_ = nullptr;
  ImFont* font_36_ = nullptr;
  ImFont* font_48_ = nullptr;
  ImTextureID logo_texture_id_ = 0;
  ImTextureID cmd_texture_id_ = 0;
};

}  // namespace gui
}  // namespace viewer
}  // namespace vkgs

#endif  // VKGS_VIEWER_GUI_TITLE_SCREEN_H

