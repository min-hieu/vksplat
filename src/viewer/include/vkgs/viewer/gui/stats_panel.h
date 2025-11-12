#ifndef VKGS_VIEWER_GUI_STATS_PANEL_H
#define VKGS_VIEWER_GUI_STATS_PANEL_H

#include <string>
#include <vector>
#include <cstdint>

struct SDL_Window;

namespace vkgs {
namespace viewer {
namespace gui {

class StatsPanel {
 public:
  StatsPanel();
  ~StatsPanel();

  void Initialize(SDL_Window* window);
  void RenderUI(bool& stats_panel_open,
                const std::vector<float>& frame_times_ms, float current_frame_time_ms);
  bool HandleClick(int x, int y, int width, int height, bool& stats_panel_open);

 private:
  SDL_Window* window_ = nullptr;
};

}  // namespace gui
}  // namespace viewer
}  // namespace vkgs

#endif  // VKGS_VIEWER_GUI_STATS_PANEL_H

