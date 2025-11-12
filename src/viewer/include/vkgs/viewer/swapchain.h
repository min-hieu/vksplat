#ifndef VKGS_VIEWER_SWAPCHAIN_H
#define VKGS_VIEWER_SWAPCHAIN_H

#include <memory>
#include <vector>

#include "volk.h"

namespace vkgs {
namespace gpu {
class Device;
}

namespace viewer {

class Swapchain {
 public:
  Swapchain(std::shared_ptr<gpu::Device> device, VkSurfaceKHR surface, uint32_t width, uint32_t height, bool vsync = true);
  ~Swapchain();

  VkSwapchainKHR handle() const { return swapchain_; }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  VkFormat format() const { return format_; }
  uint32_t image_count() const { return static_cast<uint32_t>(images_.size()); }
  VkImage image(uint32_t index) const { return images_[index]; }
  VkImageView image_view(uint32_t index) const { return image_views_[index]; }

  bool AcquireNextImage(VkSemaphore semaphore, uint32_t* image_index);
  void Present(VkQueue queue, uint32_t image_index, VkSemaphore wait_semaphore);
  bool ShouldRecreate() const;
  void Recreate(uint32_t width, uint32_t height);

 private:
  std::shared_ptr<gpu::Device> device_;
  VkSurfaceKHR surface_;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  uint32_t width_;
  uint32_t height_;
  VkFormat format_ = VK_FORMAT_B8G8R8A8_UNORM;
  VkPresentModeKHR present_mode_;
  std::vector<VkImage> images_;
  std::vector<VkImageView> image_views_;
  bool vsync_;
};

}  // namespace viewer
}  // namespace vkgs

#endif  // VKGS_VIEWER_SWAPCHAIN_H


