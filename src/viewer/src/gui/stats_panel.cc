#include "vkgs/viewer/gui/stats_panel.h"

#include "imgui.h"
#include "imgui_impl_sdl3.h"

#include <algorithm>
#include <cmath>

namespace vkgs {
namespace viewer {
namespace gui {

StatsPanel::StatsPanel() {
  // ImGui context is created by GUI class, we just use it
}

StatsPanel::~StatsPanel() = default;

void StatsPanel::Initialize(SDL_Window* window) {
  window_ = window;
}

void StatsPanel::RenderUI(bool& stats_panel_open,
                          const std::vector<float>& frame_times_ms, float current_frame_time_ms) {
  if (!window_) return;

  ImGuiIO& io = ImGui::GetIO();

  // Set initial position only on first use (allows window to be movable)
  ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 610.0f, 10.0f), ImGuiCond_FirstUseEver);

  // Set window size
  ImGui::SetNextWindowSize(ImVec2(600.0f, 450.0f), ImGuiCond_FirstUseEver);

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar;

  // CRITICAL: Always call Begin() to let ImGui manage window state properly
  // Pass &stats_panel_open so ImGui can set it to false when window is closed
  // CRITICAL: Always call End() after Begin(), even if Begin() returns false
  // This is a known ImGui requirement to properly clean up window state
  bool begin_result = ImGui::Begin("Statistics", &stats_panel_open, flags);

  if (begin_result) {
    // Latency graph
    if (!frame_times_ms.empty()) {
      // Calculate min/max for graph scaling
      float min_time = *std::min_element(frame_times_ms.begin(), frame_times_ms.end());
      float max_time = *std::max_element(frame_times_ms.begin(), frame_times_ms.end());

      // Ensure valid range
      if (max_time <= min_time) {
        max_time = min_time + 1.0f;
      }

      // Large graph size
      ImVec2 graph_size(560.0f, 350.0f);

      // Title
      ImGui::Text("Frame Latency");
      ImGui::Spacing();

      // Save cursor position before graph
      ImVec2 cursor_before = ImGui::GetCursorPos();

      // Draw the graph (offset to the right to make room for y-axis label)
      ImGui::SetCursorPosX(60.0f);  // Leave space for y-axis label
      ImGui::PlotLines("##LatencyGraph",
                      frame_times_ms.data(),
                      static_cast<int>(frame_times_ms.size()),
                      0,
                      nullptr,
                      min_time,
                      max_time,
                      graph_size);

      // Add y-axis label on the left, vertically centered with the graph
      float graph_center_y = cursor_before.y + graph_size.y * 0.5f;
      ImGui::SetCursorPos(ImVec2(10.0f, graph_center_y - 20.0f));
      ImGui::Text("Latency\n(ms)");

      // Reset cursor to after the graph
      ImGui::SetCursorPos(ImVec2(cursor_before.x, cursor_before.y + graph_size.y + 10.0f));

      // Show current min/max values
      ImGui::Text("Min: %.2f ms  |  Max: %.2f ms", min_time, max_time);
    } else {
      ImGui::Text("No data available");
    }
  }

  // CRITICAL: Always call End() after Begin(), regardless of Begin() return value
  // This is required by ImGui to properly clean up window state and prevent crashes
  ImGui::End();
}

bool StatsPanel::HandleClick(int x, int y, int width, int height, bool& stats_panel_open) {
  // ImGui handles its own input through ImGui_ImplSDL3_NewFrame
  // This function is kept for compatibility but ImGui handles clicks internally
  return false;
}

}  // namespace gui
}  // namespace viewer
}  // namespace vkgs
