#include "graphics_storage.h"

#include "vkgs/gpu/image.h"

namespace vkgs {
namespace core {

GraphicsStorage::GraphicsStorage(std::shared_ptr<gpu::Device> device) : device_(device) {}

GraphicsStorage::~GraphicsStorage() {}

void GraphicsStorage::Update(uint32_t width, uint32_t height) {
  if (width_ != width || height_ != height) {
    // Only create images if dimensions are valid
    if (width > 0 && height > 0) {
      image_ = gpu::Image::Create(device_, VK_FORMAT_R16G16B16A16_SFLOAT, width, height,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
      image_u8_ = gpu::Image::Create(
          device_, VK_FORMAT_R8G8B8A8_UNORM, width, height,
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
      depth_image_ = gpu::Image::Create(device_, VK_FORMAT_D32_SFLOAT, width, height,
                                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    } else {
      // Reset to null if dimensions are invalid
      image_.reset();
      image_u8_.reset();
      depth_image_.reset();
    }

    width_ = width;
    height_ = height;
  }
}

}  // namespace core
}  // namespace vkgs