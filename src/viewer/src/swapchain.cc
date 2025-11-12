#include "vkgs/viewer/swapchain.h"

#include <iostream>

#include "vkgs/gpu/device.h"

namespace vkgs {
namespace viewer {

Swapchain::Swapchain(std::shared_ptr<gpu::Device> device, VkSurfaceKHR surface, uint32_t width, uint32_t height, bool vsync)
    : device_(device), surface_(surface), width_(width), height_(height), vsync_(vsync) {
  VkPhysicalDevice physical_device = device_->physical_device();
  VkInstance instance = device_->instance();

  // Get surface capabilities
  VkSurfaceCapabilitiesKHR surface_capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface_, &surface_capabilities);

  // Choose present mode
  uint32_t present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface_, &present_mode_count, nullptr);
  std::vector<VkPresentModeKHR> present_modes(present_mode_count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface_, &present_mode_count, present_modes.data());

  present_mode_ = VK_PRESENT_MODE_FIFO_KHR;  // Always supported
  if (!vsync) {
    for (const auto& mode : present_modes) {
      if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
        present_mode_ = VK_PRESENT_MODE_MAILBOX_KHR;
        break;
      }
    }
  }

  // Get surface format
  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface_, &format_count, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(format_count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface_, &format_count, formats.data());

  format_ = VK_FORMAT_B8G8R8A8_UNORM;
  VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  for (const auto& fmt : formats) {
    if (fmt.format == VK_FORMAT_B8G8R8A8_UNORM && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      format_ = fmt.format;
      color_space = fmt.colorSpace;
      break;
    }
  }

  // Create swapchain
  VkSwapchainCreateInfoKHR swapchain_info = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  swapchain_info.surface = surface_;
  swapchain_info.minImageCount = 3;
  swapchain_info.imageFormat = format_;
  swapchain_info.imageColorSpace = color_space;
  swapchain_info.imageExtent = {width_, height_};
  swapchain_info.imageArrayLayers = 1;
  swapchain_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  swapchain_info.preTransform = surface_capabilities.currentTransform;
  swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchain_info.presentMode = present_mode_;
  swapchain_info.clipped = VK_TRUE;

  if (vkCreateSwapchainKHR(device_->device(), &swapchain_info, nullptr, &swapchain_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create swapchain");
  }

  // Get swapchain images
  uint32_t image_count;
  vkGetSwapchainImagesKHR(device_->device(), swapchain_, &image_count, nullptr);
  images_.resize(image_count);
  vkGetSwapchainImagesKHR(device_->device(), swapchain_, &image_count, images_.data());

  // Create image views
  image_views_.resize(image_count);
  for (uint32_t i = 0; i < image_count; ++i) {
    VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = images_[i];
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format_;
    view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device_->device(), &view_info, nullptr, &image_views_[i]) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create swapchain image view");
    }
  }
}

Swapchain::~Swapchain() {
  VkDevice device = device_->device();
  for (auto image_view : image_views_) {
    vkDestroyImageView(device, image_view, nullptr);
  }
  if (swapchain_ != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device, swapchain_, nullptr);
  }
}

bool Swapchain::AcquireNextImage(VkSemaphore semaphore, uint32_t* image_index) {
  VkResult result = vkAcquireNextImageKHR(device_->device(), swapchain_, UINT64_MAX, semaphore, VK_NULL_HANDLE, image_index);
  return result == VK_SUCCESS;
}

void Swapchain::Present(VkQueue queue, uint32_t image_index, VkSemaphore wait_semaphore) {
  VkPresentInfoKHR present_info = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &wait_semaphore;
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &swapchain_;
  present_info.pImageIndices = &image_index;

  vkQueuePresentKHR(queue, &present_info);
}

bool Swapchain::ShouldRecreate() const {
  VkSurfaceCapabilitiesKHR surface_capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_->physical_device(), surface_, &surface_capabilities);
  return surface_capabilities.currentExtent.width != width_ || surface_capabilities.currentExtent.height != height_;
}

void Swapchain::Recreate(uint32_t width, uint32_t height) {
  VkDevice device = device_->device();

  // Destroy old image views
  for (auto image_view : image_views_) {
    vkDestroyImageView(device, image_view, nullptr);
  }
  image_views_.clear();
  images_.clear();

  // Destroy old swapchain
  if (swapchain_ != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device, swapchain_, nullptr);
  }

  width_ = width;
  height_ = height;

  // Recreate (similar to constructor)
  VkPhysicalDevice physical_device = device_->physical_device();
  VkSurfaceCapabilitiesKHR surface_capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface_, &surface_capabilities);

  VkSwapchainCreateInfoKHR swapchain_info = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  swapchain_info.surface = surface_;
  swapchain_info.minImageCount = 3;
  swapchain_info.imageFormat = format_;
  swapchain_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  swapchain_info.imageExtent = {width_, height_};
  swapchain_info.imageArrayLayers = 1;
  swapchain_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  swapchain_info.preTransform = surface_capabilities.currentTransform;
  swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchain_info.presentMode = present_mode_;
  swapchain_info.clipped = VK_TRUE;

  if (vkCreateSwapchainKHR(device, &swapchain_info, nullptr, &swapchain_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to recreate swapchain");
  }

  // Get swapchain images
  uint32_t image_count;
  vkGetSwapchainImagesKHR(device, swapchain_, &image_count, nullptr);
  images_.resize(image_count);
  vkGetSwapchainImagesKHR(device, swapchain_, &image_count, images_.data());

  // Create image views
  image_views_.resize(image_count);
  for (uint32_t i = 0; i < image_count; ++i) {
    VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = images_[i];
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format_;
    view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &view_info, nullptr, &image_views_[i]) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create swapchain image view");
    }
  }
}

}  // namespace viewer
}  // namespace vkgs


