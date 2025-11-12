#ifndef VKGS_VIEWER_GUI_H
#define VKGS_VIEWER_GUI_H

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <memory>

#include "volk.h"
#include "vk_mem_alloc.h"

struct SDL_Window;

// Include ImGui for ImTextureID
#include "imgui.h"

namespace vkgs {
namespace gpu {
class Device;
}
namespace viewer {
namespace gui {
class TitleScreen;
class StatsPanel;
}

class GUI {
 public:
  GUI(const std::string& assets_path);
  ~GUI();

  void Initialize(SDL_Window* window);

  // Vulkan initialization
  void InitializeVulkan(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, uint32_t queue_family_index, VkFormat swapchain_format, VkRenderPass render_pass, VmaAllocator allocator, std::shared_ptr<vkgs::gpu::Device> gpu_device);
  void ShutdownVulkan();
  void RecreateFonts();
  void UpdateRenderPass(VkRenderPass render_pass);

  // Texture loading (for SVG images)
  ImTextureID LoadSVGTexture(const std::string& svg_path, int width, int height);

  // Rendering methods - now using Vulkan command buffer
  void RenderTitleScreen(VkCommandBuffer command_buffer, VkFramebuffer framebuffer,
                         uint32_t width, uint32_t height,
                         bool& showing_title_screen, std::string& pending_ply_path,
                         std::function<std::string()> show_file_picker);
  void RenderStatsPanel(VkCommandBuffer command_buffer, VkFramebuffer framebuffer,
                        uint32_t width, uint32_t height,
                        bool showing_title_screen, bool& stats_panel_open,
                        const std::vector<float>& frame_times_ms, float current_frame_time_ms);

  // Click handling
  bool HandleTitleScreenClick(int x, int y, int width, int height,
                               std::string& pending_ply_path, std::function<std::string()> show_file_picker);
  bool HandleStatsPanelClick(int x, int y, int width, int height, bool& stats_panel_open);

 private:
  std::string assets_path_;
  std::unique_ptr<gui::TitleScreen> title_screen_;
  std::unique_ptr<gui::StatsPanel> stats_panel_;

  // Vulkan state
  bool vulkan_initialized_ = false;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  std::shared_ptr<vkgs::gpu::Device> gpu_device_;

  // Texture management
  struct TextureInfo {
    VkImage image = VK_NULL_HANDLE;
    VkImageView image_view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    ImTextureID texture_id = 0;
  };
  std::vector<std::unique_ptr<TextureInfo>> loaded_textures_;
};  // class GUI

}  // namespace viewer
}  // namespace vkgs

#endif  // VKGS_VIEWER_GUI_H
