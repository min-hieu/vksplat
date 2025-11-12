#include "vkgs/viewer/gui.h"
#include "vkgs/viewer/gui/title_screen.h"
#include "vkgs/viewer/gui/stats_panel.h"

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"

#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h"
#define NANOSVGRAST_IMPLEMENTATION
#include "nanosvgrast.h"

#include "vkgs/gpu/device.h"
#include "vkgs/gpu/fence.h"
#include "vkgs/gpu/command.h"
#include "vkgs/gpu/queue.h"

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

namespace vkgs {
namespace viewer {

GUI::GUI(const std::string& assets_path) : assets_path_(assets_path) {
  // Create shared ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  // Style settings
  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 4.0f;
  style.FrameRounding = 4.0f;
  style.ScrollbarRounding = 4.0f;
  style.WindowPadding = ImVec2(8.0f, 8.0f);
  style.ItemSpacing = ImVec2(8.0f, 4.0f);
  style.ScrollbarSize = 20.0f;
  style.GrabMinSize = 20.0f;

  // Initialize title screen and stats panel (both will use the shared context)
  title_screen_ = std::make_unique<gui::TitleScreen>(assets_path_);
  stats_panel_ = std::make_unique<gui::StatsPanel>();

  vulkan_initialized_ = false;
}

GUI::~GUI() {
  ShutdownVulkan();

  // Clean up ImGui context
  if (ImGui::GetCurrentContext()) {
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
  }
}

void GUI::Initialize(SDL_Window* window) {
  // Initialize ImGui SDL3 backend
  ImGui_ImplSDL3_InitForVulkan(window);

  // Initialize both components
  if (title_screen_) {
    title_screen_->Initialize(window);
  }
  if (stats_panel_) {
    stats_panel_->Initialize(window);
  }
}

void GUI::InitializeVulkan(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, uint32_t queue_family_index, VkFormat swapchain_format, VkRenderPass render_pass, VmaAllocator allocator, std::shared_ptr<vkgs::gpu::Device> gpu_device) {
  if (vulkan_initialized_) {
    ShutdownVulkan();
  }

  device_ = device;
  queue_ = queue;
  swapchain_format_ = swapchain_format;
  render_pass_ = render_pass;
  allocator_ = allocator;
  gpu_device_ = gpu_device;

  // Initialize ImGui Vulkan backend
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.ApiVersion = VK_API_VERSION_1_0;
  init_info.Instance = instance;
  init_info.PhysicalDevice = physical_device;
  init_info.Device = device;
  init_info.QueueFamily = queue_family_index;
  init_info.Queue = queue;
  init_info.PipelineCache = VK_NULL_HANDLE;
  init_info.DescriptorPool = VK_NULL_HANDLE;  // Will create our own
  init_info.DescriptorPoolSize = 0;  // We'll create our own pool
  init_info.MinImageCount = 3;
  init_info.ImageCount = 3;
  init_info.UseDynamicRendering = false;  // We're using traditional render passes
  init_info.Allocator = nullptr;
  init_info.CheckVkResultFn = nullptr;
  init_info.MinAllocationSize = 0;

  // Setup pipeline info for main viewport
  init_info.PipelineInfoMain.RenderPass = render_pass;
  init_info.PipelineInfoMain.Subpass = 0;
  init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

  // Setup pipeline info for viewports (same as main)
  init_info.PipelineInfoForViewports = init_info.PipelineInfoMain;

  // Create descriptor pool for ImGui
  VkDescriptorPoolSize pool_sizes[] = {
    { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
    { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
    { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
    { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
    { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
    { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
  };

  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 1000;
  pool_info.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]);
  pool_info.pPoolSizes = pool_sizes;

  if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create ImGui descriptor pool");
  }

  init_info.DescriptorPool = descriptor_pool_;

  // Load Vulkan functions (required when using volk or VK_NO_PROTOTYPES)
  ImGui_ImplVulkan_LoadFunctions(init_info.ApiVersion, [](const char* function_name, void* user_data) {
    return vkGetInstanceProcAddr(*(VkInstance*)user_data, function_name);
  }, &instance);

  ImGui_ImplVulkan_Init(&init_info);

  vulkan_initialized_ = true;

  // Note: Fonts will be uploaded automatically by ImGui_ImplVulkan_NewFrame() on first call
}

void GUI::ShutdownVulkan() {
  if (!vulkan_initialized_) return;

  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);

    // Clean up loaded textures
    for (auto& texture : loaded_textures_) {
      if (texture) {
        if (texture->sampler != VK_NULL_HANDLE) {
          vkDestroySampler(device_, texture->sampler, nullptr);
        }
        if (texture->image_view != VK_NULL_HANDLE) {
          vkDestroyImageView(device_, texture->image_view, nullptr);
        }
        if (texture->image != VK_NULL_HANDLE && allocator_ != VK_NULL_HANDLE) {
          vmaDestroyImage(allocator_, texture->image, texture->allocation);
        }
      }
    }
    loaded_textures_.clear();

    ImGui_ImplVulkan_Shutdown();

    if (descriptor_pool_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
      descriptor_pool_ = VK_NULL_HANDLE;
    }
  }

  vulkan_initialized_ = false;
  allocator_ = VK_NULL_HANDLE;
  gpu_device_.reset();
}

void GUI::RecreateFonts() {
  if (!vulkan_initialized_) return;

  // Upload font textures
  // Note: In newer ImGui versions, this creates its own command buffer internally
  // Fonts are created automatically by ImGui_ImplVulkan_NewFrame() on first call
}

void GUI::UpdateRenderPass(VkRenderPass render_pass) {
  if (!vulkan_initialized_) return;

  // Update the render pass reference
  // Note: ImGui doesn't have a direct function to update the render pass,
  // but as long as the render pass is compatible (same format, same subpass),
  // we can just update our internal reference
  render_pass_ = render_pass;
}

void GUI::RenderTitleScreen(VkCommandBuffer command_buffer, VkFramebuffer framebuffer,
                            uint32_t width, uint32_t height,
                            bool& showing_title_screen, std::string& pending_ply_path,
                            std::function<std::string()> show_file_picker) {
  if (!vulkan_initialized_ || !title_screen_ || framebuffer == VK_NULL_HANDLE) return;

  // Set display size BEFORE starting the frame
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
  io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

  // Start ImGui frame (this will use the DisplaySize we just set)
  ImGui_ImplSDL3_NewFrame();
  ImGui_ImplVulkan_NewFrame();  // This automatically handles font texture creation on first call
  ImGui::NewFrame();

  // Render title screen UI
  auto load_texture = [this](const std::string& path, int w, int h) -> ImTextureID {
    return LoadSVGTexture(path, w, h);
  };
  title_screen_->RenderUI(pending_ply_path, show_file_picker, load_texture);

  // Render ImGui to Vulkan
  ImGui::Render();
  ImDrawData* draw_data = ImGui::GetDrawData();

  if (draw_data) {
    // Begin render pass with framebuffer (required for ImGui Vulkan backend)
    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = render_pass_;
    render_pass_info.framebuffer = framebuffer;
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = {width, height};

    // Clear to dark gray for title screen
    VkClearValue clear_value = {};
    clear_value.color = {{0.1f, 0.1f, 0.1f, 1.0f}};  // Dark gray
    render_pass_info.clearValueCount = 1;
    render_pass_info.pClearValues = &clear_value;

    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    // Set viewport and scissor to cover the full screen
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {width, height};
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    // Record ImGui draw commands (ImGui handles scaling internally)
    ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);

    vkCmdEndRenderPass(command_buffer);
  }
}

void GUI::RenderStatsPanel(VkCommandBuffer command_buffer, VkFramebuffer framebuffer,
                           uint32_t width, uint32_t height,
                           bool showing_title_screen, bool& stats_panel_open,
                           const std::vector<float>& frame_times_ms, float current_frame_time_ms) {
  if (!vulkan_initialized_ || showing_title_screen || !stats_panel_ || framebuffer == VK_NULL_HANDLE) return;

  // Set display size BEFORE starting the frame
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
  io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

  // Start ImGui frame (this will use the DisplaySize we just set)
  ImGui_ImplSDL3_NewFrame();
  ImGui_ImplVulkan_NewFrame();  // This automatically handles font texture creation on first call
  ImGui::NewFrame();

  // Render stats panel UI
  stats_panel_->RenderUI(stats_panel_open, frame_times_ms, current_frame_time_ms);

  // Render ImGui to Vulkan
  ImGui::Render();
  ImDrawData* draw_data = ImGui::GetDrawData();

  if (draw_data) {
    // Begin render pass with framebuffer (required for ImGui Vulkan backend)
    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = render_pass_;
    render_pass_info.framebuffer = framebuffer;
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = {width, height};
    render_pass_info.clearValueCount = 0;
    render_pass_info.pClearValues = nullptr;

    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    // Set viewport and scissor to cover the full screen
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {width, height};
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    // Record ImGui draw commands (ImGui handles scaling internally)
    ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);

    vkCmdEndRenderPass(command_buffer);
  }
}

bool GUI::HandleTitleScreenClick(int x, int y, int width, int height,
                                  std::string& pending_ply_path, std::function<std::string()> show_file_picker) {
  if (title_screen_) {
    return title_screen_->HandleClick(x, y, width, height, pending_ply_path, show_file_picker);
  }
  return false;
}

bool GUI::HandleStatsPanelClick(int x, int y, int width, int height, bool& stats_panel_open) {
  if (stats_panel_) {
    return stats_panel_->HandleClick(x, y, width, height, stats_panel_open);
  }
  return false;
}

ImTextureID GUI::LoadSVGTexture(const std::string& svg_path, int width, int height) {
  if (!vulkan_initialized_ || device_ == VK_NULL_HANDLE || allocator_ == VK_NULL_HANDLE || !gpu_device_) {
    return 0;
  }

  // Check if texture already loaded
  for (const auto& texture : loaded_textures_) {
    // We could cache by path, but for now just check if we have space
  }

  // Parse SVG
  NSVGimage* image = nsvgParseFromFile(svg_path.c_str(), "px", 96.0f);
  if (!image) {
    std::cerr << "Warning: Could not load SVG from " << svg_path << std::endl;
    return 0;
  }

  // Rasterize SVG to RGBA
  std::vector<unsigned char> raster(width * height * 4);
  NSVGrasterizer* rast = nsvgCreateRasterizer();
  float scale = std::min(static_cast<float>(width) / image->width, static_cast<float>(height) / image->height);
  nsvgRasterize(rast, image, 0, 0, scale, raster.data(), width, height, width * 4);
  nsvgDeleteRasterizer(rast);
  nsvgDelete(image);

  // Create texture info
  auto texture_info = std::make_unique<TextureInfo>();

  // Create Vulkan image
  VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  image_info.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo alloc_info = {};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

  if (vmaCreateImage(allocator_, &image_info, &alloc_info, &texture_info->image, &texture_info->allocation, nullptr) != VK_SUCCESS) {
    std::cerr << "Failed to create image for SVG texture" << std::endl;
    return 0;
  }

  // Create staging buffer and upload
  VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer_info.size = raster.size();
  buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo staging_alloc_info = {};
  staging_alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

  VkBuffer staging_buffer;
  VmaAllocation staging_allocation;
  if (vmaCreateBuffer(allocator_, &buffer_info, &staging_alloc_info, &staging_buffer, &staging_allocation, nullptr) != VK_SUCCESS) {
    vmaDestroyImage(allocator_, texture_info->image, texture_info->allocation);
    return 0;
  }

  // Copy raster data to staging buffer
  void* mapped;
  vmaMapMemory(allocator_, staging_allocation, &mapped);
  std::memcpy(mapped, raster.data(), raster.size());
  vmaFlushAllocation(allocator_, staging_allocation, 0, raster.size());
  vmaUnmapMemory(allocator_, staging_allocation);

  // Allocate command buffer and fence
  auto gq = gpu_device_->graphics_queue();
  auto cb = gq->AllocateCommandBuffer();
  auto fence = gpu_device_->AllocateFence();

  VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(*cb, &begin_info);

  // Transition image to transfer dst
  VkImageMemoryBarrier2 barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
  barrier.srcAccessMask = 0;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.image = texture_info->image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

  VkDependencyInfo dep_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep_info.imageMemoryBarrierCount = 1;
  dep_info.pImageMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(*cb, &dep_info);

  // Copy buffer to image
  VkBufferImageCopy region = {};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
  vkCmdCopyBufferToImage(*cb, staging_buffer, texture_info->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  // Transition image to shader read only
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  vkCmdPipelineBarrier2(*cb, &dep_info);

  vkEndCommandBuffer(*cb);

  // Submit command buffer
  VkCommandBufferSubmitInfo cmd_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cmd_info.commandBuffer = *cb;

  VkSubmitInfo2 submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit_info.commandBufferInfoCount = 1;
  submit_info.pCommandBufferInfos = &cmd_info;

  VkFence vk_fence = *fence;
  vkQueueSubmit2(queue_, 1, &submit_info, vk_fence);
  fence->Wait();

  // Clean up staging buffer
  vmaDestroyBuffer(allocator_, staging_buffer, staging_allocation);

  // Create image view
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = texture_info->image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

  if (vkCreateImageView(device_, &view_info, nullptr, &texture_info->image_view) != VK_SUCCESS) {
    vmaDestroyImage(allocator_, texture_info->image, texture_info->allocation);
    return 0;
  }

  // Create sampler
  VkSamplerCreateInfo sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.anisotropyEnable = VK_FALSE;
  sampler_info.maxAnisotropy = 1.0f;
  sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipLodBias = 0.0f;
  sampler_info.minLod = 0.0f;
  sampler_info.maxLod = 0.0f;

  if (vkCreateSampler(device_, &sampler_info, nullptr, &texture_info->sampler) != VK_SUCCESS) {
    vkDestroyImageView(device_, texture_info->image_view, nullptr);
    vmaDestroyImage(allocator_, texture_info->image, texture_info->allocation);
    return 0;
  }

  // Register with ImGui
  VkDescriptorSet descriptor_set = ImGui_ImplVulkan_AddTexture(texture_info->sampler, texture_info->image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  texture_info->texture_id = reinterpret_cast<ImTextureID>(descriptor_set);

  // Store texture info and return the texture ID
  ImTextureID result = texture_info->texture_id;
  loaded_textures_.push_back(std::move(texture_info));

  return result;
}

}  // namespace viewer
}  // namespace vkgs
