#ifndef VKGS_VIEWER_VIEWER_H
#define VKGS_VIEWER_VIEWER_H

#include <memory>
#include <string>
#include <functional>
#include <chrono>
#include <vector>

#include "volk.h"
#include "vk_mem_alloc.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "vkgs/gpu/command.h"
#include "vkgs/viewer/gui.h"

struct SDL_Window;
struct SDL_Gamepad;

namespace vkgs {
namespace core {
class Renderer;
class GaussianSplats;
}
namespace gpu {
class Device;
}

namespace viewer {

class Swapchain;

class Viewer {

 public:
  Viewer(std::shared_ptr<core::Renderer> renderer, const std::string& title = "SplatStream Viewer", uint32_t width = 1280, uint32_t height = 720);
  ~Viewer();

  void Run(const std::string& ply_path = "");
  void Close();

  SDL_Window* window() const { return window_; }
  VkSurfaceKHR surface() const { return surface_; }

 private:
  void SetupCallbacks();
  void UpdateCamera();
  void ProcessControllerInput();
  void RenderFrame();
  void RenderTitleScreen();
  void RecreateSwapchainResources();
  void CreateImGuiRenderPass(std::shared_ptr<gpu::Device> device, VkFormat swapchain_format);
  glm::mat4 GetViewMatrix() const;
  std::string ShowFilePicker();
  void WarmUpFilePicker();  // Pre-initialize file picker system for instant opening

  // Arcball helper functions
  glm::vec3 ProjectToSphere(float x, float y, float width, float height) const;
  glm::quat ComputeArcballRotation(const glm::vec3& from, const glm::vec3& to) const;

  std::shared_ptr<core::Renderer> renderer_;
  SDL_Window* window_ = nullptr;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  std::unique_ptr<Swapchain> swapchain_;

  std::shared_ptr<core::GaussianSplats> splats_;

  uint32_t width_;
  uint32_t height_;
  bool should_close_ = false;
  bool showing_title_screen_ = true;
  std::string pending_ply_path_;

  // Camera state - using arcball (quaternion-based) instead of azimuth/elevation
  float camera_distance_ = 10.0f;
  glm::quat camera_rotation_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);  // Identity quaternion
  float camera_fov_ = 45.0f;
  float camera_near_ = 0.1f;
  float camera_far_ = 1000.0f;
  float camera_center_[3] = {0.0f, 0.0f, 0.0f};

  // Arcball state
  glm::vec3 arcball_start_ = glm::vec3(0.0f, 0.0f, 1.0f);  // Initial click position on sphere
  glm::quat arcball_base_rotation_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);  // Base rotation when drag started
  bool arcball_active_ = false;  // Whether arcball is currently being dragged
  float arcball_sensitivity_ = 2.0f;  // Sensitivity multiplier for rotation

  // Input state
  bool mouse_left_pressed_ = false;
  bool mouse_right_pressed_ = false;
  double last_mouse_x_ = 0.0;
  double last_mouse_y_ = 0.0;
  double current_mouse_x_ = 0.0;
  double current_mouse_y_ = 0.0;
  float scroll_offset_ = 0.0f;

  // Controller state
  SDL_Gamepad* controller_ = nullptr;
  float controller_left_stick_x_ = 0.0f;
  float controller_left_stick_y_ = 0.0f;
  float controller_right_stick_x_ = 0.0f;
  float controller_right_stick_y_ = 0.0f;
  float controller_trigger_left_ = 0.0f;
  float controller_trigger_right_ = 0.0f;
  float controller_rotation_speed_ = 2.0f;
  float controller_pan_speed_ = 0.5f;
  float controller_move_speed_ = 1.0f;

  // Frame tracking
  uint32_t frame_counter_ = 0;
  uint32_t current_image_index_ = 0;

  // Frame profiler
  bool stats_panel_open_ = false;
  static constexpr size_t FRAME_HISTORY_SIZE = 300;  // Store last 300 frames
  std::vector<float> frame_times_ms_;  // Frame times in milliseconds
  std::chrono::high_resolution_clock::time_point last_frame_time_;
  float current_frame_time_ms_ = 0.0f;

  // Visual options
  bool visual_panel_open_ = false;
  bool visualize_depth_ = false;
  bool depth_auto_range_ = false;
  float depth_z_min_ = 22.0f;
  float depth_z_max_ = 50.0f;

  // Binary semaphores for swapchain (one per swapchain image)
  std::vector<VkSemaphore> image_acquired_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> render_finished_fences_;
  // Track which image was last using each semaphore (to wait for fence before reuse)
  std::vector<uint32_t> semaphore_last_image_;

  // Track which swapchain images have been presented (to know their layout)
  std::vector<bool> image_has_been_presented_;
  std::vector<std::shared_ptr<gpu::Command>> command_buffers_;

  // Reusable buffers for rendering
  std::vector<uint8_t> image_data_;
  VkBuffer staging_buffer_ = VK_NULL_HANDLE;
  VmaAllocation staging_allocation_ = VK_NULL_HANDLE;
  size_t staging_size_ = 0;

  // GUI
  std::unique_ptr<GUI> gui_;
  VkRenderPass imgui_render_pass_ = VK_NULL_HANDLE;
  std::vector<VkFramebuffer> imgui_framebuffers_;

  std::string assets_path_;
};

}  // namespace viewer
}  // namespace vkgs

#endif  // VKGS_VIEWER_VIEWER_H

