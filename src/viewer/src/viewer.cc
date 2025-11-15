#include "vkgs/viewer/viewer.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <chrono>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_properties.h>
#ifdef __APPLE__
#include <SDL3/SDL_video.h>
#endif

#include "imgui.h"
#include "imgui_impl_sdl3.h"

#ifdef __APPLE__
#include <Cocoa/Cocoa.h>
#include <dispatch/dispatch.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "vk_mem_alloc.h"


#include "vkgs/viewer/swapchain.h"
#include "vkgs/gpu/device.h"
#include "vkgs/viewer/gui.h"
#include "vkgs/core/renderer.h"
#include "vkgs/core/gaussian_splats.h"
#include "vkgs/core/rendered_image.h"
#include "vkgs/core/draw_options.h"
#include "vkgs/gpu/device.h"
#include "vkgs/gpu/queue.h"
#include "vkgs/gpu/command.h"
#include "vkgs/gpu/image.h"
#include "vkgs/gpu/buffer.h"
#include "vkgs/gpu/semaphore.h"
#include "vkgs/gpu/fence.h"

namespace vkgs {
namespace viewer {

namespace {

struct PlyBounds {
  glm::vec3 min;
  glm::vec3 max;
  bool valid = false;
};

std::optional<PlyBounds> ComputeBoundsFromPly(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Warning: Unable to open PLY file for bounds: " << path << std::endl;
    return std::nullopt;
  }

  std::string line;
  bool in_vertex_element = false;
  size_t vertex_count = 0;
  size_t property_count = 0;

  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();

    if (line.rfind("element vertex", 0) == 0) {
      in_vertex_element = true;
      std::string count_str = line.substr(std::strlen("element vertex"));
      try {
        vertex_count = static_cast<size_t>(std::stoull(count_str));
      } catch (...) {
        vertex_count = 0;
      }
    } else if (line.rfind("element ", 0) == 0) {
      in_vertex_element = false;
    } else if (in_vertex_element && line.rfind("property", 0) == 0) {
      ++property_count;
    } else if (line == "end_header") {
      break;
    }
  }

  if (vertex_count == 0 || property_count < 3) {
    std::cerr << "Warning: Invalid PLY header for bounds computation" << std::endl;
    return std::nullopt;
  }

  const size_t vertex_stride = property_count * sizeof(float);
  PlyBounds bounds;
  bounds.min = glm::vec3(std::numeric_limits<float>::max());
  bounds.max = glm::vec3(std::numeric_limits<float>::lowest());

  for (size_t i = 0; i < vertex_count; ++i) {
    float xyz[3];
    file.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
    if (!file) break;

    glm::vec3 position(xyz[0], xyz[1], xyz[2]);
    bounds.min.x = std::min(bounds.min.x, position.x);
    bounds.min.y = std::min(bounds.min.y, position.y);
    bounds.min.z = std::min(bounds.min.z, position.z);
    bounds.max.x = std::max(bounds.max.x, position.x);
    bounds.max.y = std::max(bounds.max.y, position.y);
    bounds.max.z = std::max(bounds.max.z, position.z);

    if (vertex_stride > sizeof(xyz)) {
      file.seekg(static_cast<std::streamoff>(vertex_stride - sizeof(xyz)), std::ios::cur);
    }
  }

  if (!file.good() && !file.eof()) {
    std::cerr << "Warning: Failed to read PLY data for bounds" << std::endl;
    return std::nullopt;
  }

  bounds.valid = true;
  return bounds;
}


}  // namespace

// Event handling will be done in Run() method using SDL3 event loop

Viewer::Viewer(std::shared_ptr<core::Renderer> renderer, const std::string& title, uint32_t width, uint32_t height)
    : renderer_(renderer), width_(width), height_(height) {
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
    throw std::runtime_error(std::string("Failed to initialize SDL3: ") + SDL_GetError());
  }

  // Open first available game controller
  SDL_UpdateGamepads();
  int num_gamepads = 0;
  SDL_JoystickID* gamepad_ids = SDL_GetGamepads(&num_gamepads);
  if (gamepad_ids && num_gamepads > 0) {
    for (int i = 0; i < num_gamepads; ++i) {
      if (SDL_IsGamepad(gamepad_ids[i])) {
        controller_ = SDL_OpenGamepad(gamepad_ids[i]);
        if (controller_) {
          std::cout << "Game controller connected: " << SDL_GetGamepadName(controller_) << std::endl;
          break;
        }
      }
    }
    SDL_free(gamepad_ids);
  }

  window_ = SDL_CreateWindow(title.c_str(), width, height, SDL_WINDOW_VULKAN);
  if (!window_) {
    SDL_Quit();
    throw std::runtime_error(std::string("Failed to create SDL3 window: ") + SDL_GetError());
  }

  auto device = renderer_->device();
  VkInstance instance = device->instance();
  VkDevice vk_device = device->device();

  if (!SDL_Vulkan_CreateSurface(window_, instance, nullptr, &surface_)) {
    SDL_DestroyWindow(window_);
    SDL_Quit();
    throw std::runtime_error(std::string("Failed to create Vulkan surface: ") + SDL_GetError());
  }

  swapchain_ = std::make_unique<Swapchain>(device, surface_, width, height, true);

  // Create binary semaphores for swapchain operations
  uint32_t image_count = swapchain_->image_count();
  image_acquired_semaphores_.resize(image_count);
  render_finished_semaphores_.resize(image_count);
  render_finished_fences_.resize(image_count);
  semaphore_last_image_.resize(image_count, UINT32_MAX);  // Initialize to invalid index

  VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (uint32_t i = 0; i < image_count; ++i) {
    vkCreateSemaphore(vk_device, &semaphore_info, nullptr, &image_acquired_semaphores_[i]);
    vkCreateSemaphore(vk_device, &semaphore_info, nullptr, &render_finished_semaphores_[i]);
    vkCreateFence(vk_device, &fence_info, nullptr, &render_finished_fences_[i]);
  }
  image_has_been_presented_.assign(image_count, false);
  command_buffers_.resize(image_count);

  // Determine assets path (relative to executable)
  // Try multiple possible paths
  std::vector<std::string> possible_paths = {
    "inspector/assets",
    "../inspector/assets",
    "../../inspector/assets",
    "../../../inspector/assets",
    "../../../../inspector/assets"
  };

  for (const auto& path : possible_paths) {
    if (std::filesystem::exists(path + "/Ubuntu_Mono/UbuntuMono-Regular.ttf")) {
      assets_path_ = path;
      break;
    }
  }

  // Initialize GUI
  if (assets_path_.empty()) {
    std::cerr << "Warning: Could not find assets directory. Title screen may not render correctly." << std::endl;
  }
  gui_ = std::make_unique<GUI>(assets_path_);
  gui_->Initialize(window_);

  // Create render pass for ImGui (compatible with swapchain format)
  CreateImGuiRenderPass(device, swapchain_->format());

  // Initialize ImGui Vulkan backend
  auto gq = device->graphics_queue();
  VkPhysicalDevice physical_device = device->physical_device();
  gui_->InitializeVulkan(instance, physical_device, vk_device, *gq, gq->family_index(), swapchain_->format(), imgui_render_pass_, device->allocator(), device);

  // Initialize frame profiler
  frame_times_ms_.reserve(FRAME_HISTORY_SIZE);
  last_frame_time_ = std::chrono::high_resolution_clock::now();

  // Warm up file picker system for instant opening
  WarmUpFilePicker();
}

Viewer::~Viewer() {
  // Close game controller
  if (controller_) {
    SDL_CloseGamepad(controller_);
    controller_ = nullptr;
  }

  auto device = renderer_->device();
  VkDevice vk_device = device->device();

  // Wait for all operations to complete
  device->WaitIdle();

  // Shutdown GUI (this will clean up ImGui Vulkan resources)
  if (gui_) {
    gui_->ShutdownVulkan();
    gui_.reset();
  }

  // Clean up ImGui framebuffers (must be done before destroying render pass)
  for (auto framebuffer : imgui_framebuffers_) {
    if (framebuffer != VK_NULL_HANDLE) {
      vkDestroyFramebuffer(vk_device, framebuffer, nullptr);
    }
  }
  imgui_framebuffers_.clear();

  // Clean up ImGui render pass
  if (imgui_render_pass_ != VK_NULL_HANDLE) {
    vkDestroyRenderPass(vk_device, imgui_render_pass_, nullptr);
    imgui_render_pass_ = VK_NULL_HANDLE;
  }

  if (staging_buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(device->allocator(), staging_buffer_, staging_allocation_);
    staging_buffer_ = VK_NULL_HANDLE;
    staging_allocation_ = VK_NULL_HANDLE;
    staging_size_ = 0;
  }

  command_buffers_.clear();

  // Destroy semaphores and fences
  for (auto semaphore : image_acquired_semaphores_) {
    vkDestroySemaphore(vk_device, semaphore, nullptr);
  }
  for (auto semaphore : render_finished_semaphores_) {
    vkDestroySemaphore(vk_device, semaphore, nullptr);
  }
  for (auto fence : render_finished_fences_) {
    vkDestroyFence(vk_device, fence, nullptr);
  }

  if (swapchain_) {
    swapchain_.reset();
  }
  if (surface_ != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(device->instance(), surface_, nullptr);
  }
  if (window_) {
    SDL_DestroyWindow(window_);
  }
  SDL_Quit();
}

// Callbacks are now handled in the Run() method via SDL3 event loop

// Project mouse coordinates to a sphere for arcball control
glm::vec3 Viewer::ProjectToSphere(float x, float y, float width, float height) const {
  // Normalize coordinates to [-1, 1] range
  float nx = (2.0f * x / width) - 1.0f;
  // Flip Y axis: screen Y=0 is top, we want normalized Y=-1 at top
  float ny = 1.0f - (2.0f * y / height);

  // Project to sphere (or hyperbola if outside sphere)
  float length_squared = nx * nx + ny * ny;
  float z;

  if (length_squared <= 1.0f) {
    // Inside sphere: z = sqrt(1 - x^2 - y^2)
    z = std::sqrt(1.0f - length_squared);
  } else {
    // Outside sphere: project to hyperbola
    float length = std::sqrt(length_squared);
    nx /= length;
    ny /= length;
    z = 0.0f;
  }

  return glm::vec3(nx, ny, z);
}

// Compute rotation quaternion from two points on sphere
glm::quat Viewer::ComputeArcballRotation(const glm::vec3& from, const glm::vec3& to) const {
  // Normalize input vectors
  glm::vec3 from_norm = glm::normalize(from);
  glm::vec3 to_norm = glm::normalize(to);

  // Compute rotation axis (cross product)
  // Use cross(from, to) and negate Y component of axis to fix vertical inversion
  // This ensures dragging right rotates right, dragging down rotates down
  glm::vec3 axis = glm::cross(from_norm, to_norm);
  // Negate Y component to fix vertical axis inversion
  axis.y = -axis.y;
  float axis_length = glm::length(axis);

  // If vectors are parallel or opposite, return identity
  if (axis_length < 1e-6f) {
    return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }

  axis = glm::normalize(axis);

  // Compute rotation angle (dot product gives cos(angle))
  float dot = glm::dot(from_norm, to_norm);
  float angle = std::acos(std::max(-1.0f, std::min(1.0f, dot)));

  // Apply sensitivity multiplier to make rotation more responsive
  angle *= arcball_sensitivity_;

  // Create quaternion from axis-angle representation
  return glm::angleAxis(angle, axis);
}

glm::mat4 Viewer::GetViewMatrix() const {
  // Apply rotation quaternion to default camera position
  // Default: camera looks along -Z axis, positioned at (0, 0, distance)
  glm::vec3 default_eye = glm::vec3(0.0f, 0.0f, camera_distance_);
  glm::vec3 default_forward = glm::vec3(0.0f, 0.0f, -1.0f);
  glm::vec3 default_up = glm::vec3(0.0f, 1.0f, 0.0f);

  // Rotate camera position and up vector by quaternion
  glm::mat4 rotation_matrix = glm::mat4_cast(camera_rotation_);
  glm::vec3 eye = glm::vec3(rotation_matrix * glm::vec4(default_eye, 1.0f));
  glm::vec3 up = glm::vec3(rotation_matrix * glm::vec4(default_up, 0.0f));

  // Translate to camera center
  glm::vec3 center(camera_center_[0], camera_center_[1], camera_center_[2]);
  eye = center + eye;

  return glm::lookAt(eye, center, up);
}

void Viewer::ProcessControllerInput() {
  if (!controller_) return;

  // Get delta time for frame-rate independent movement
  auto current_time = std::chrono::high_resolution_clock::now();
  float delta_time = std::chrono::duration<float>(current_time - last_frame_time_).count();
  delta_time = std::min(delta_time, 0.1f);  // Cap at 100ms to avoid large jumps

  // Apply deadzone to sticks (ignore small movements)
  const float deadzone = 0.15f;

  // Right joystick: Camera rotation (first-person style)
  float right_x = std::abs(controller_right_stick_x_) > deadzone ? controller_right_stick_x_ : 0.0f;
  float right_y = std::abs(controller_right_stick_y_) > deadzone ? controller_right_stick_y_ : 0.0f;

  if (std::abs(right_x) > deadzone || std::abs(right_y) > deadzone) {
    // Yaw (horizontal rotation around Y axis)
    float yaw_delta = -right_x * controller_rotation_speed_ * delta_time;
    glm::quat yaw_rotation = glm::angleAxis(yaw_delta, glm::vec3(0.0f, 1.0f, 0.0f));

    // Pitch (vertical rotation around local X axis)
    float pitch_delta = -right_y * controller_rotation_speed_ * delta_time;
    glm::mat4 rotation_matrix = glm::mat4_cast(camera_rotation_);
    glm::vec3 local_right = glm::vec3(rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0]);
    glm::quat pitch_rotation = glm::angleAxis(pitch_delta, local_right);

    // Apply rotations: yaw first (global), then pitch (local)
    camera_rotation_ = pitch_rotation * camera_rotation_ * yaw_rotation;
  }

  // Left joystick: Panning (camera translation)
  float left_x = std::abs(controller_left_stick_x_) > deadzone ? controller_left_stick_x_ : 0.0f;
  float left_y = std::abs(controller_left_stick_y_) > deadzone ? controller_left_stick_y_ : 0.0f;

  if (std::abs(left_x) > deadzone || std::abs(left_y) > deadzone) {
    glm::mat4 view = GetViewMatrix();
    glm::vec3 right = glm::vec3(view[0][0], view[1][0], view[2][0]);
    glm::vec3 up = glm::vec3(view[0][1], view[1][1], view[2][1]);

    float pan_speed = controller_pan_speed_ * camera_distance_ * delta_time;
    camera_center_[0] += right.x * left_x * pan_speed;
    camera_center_[1] += right.y * left_x * pan_speed;
    camera_center_[2] += right.z * left_x * pan_speed;
    camera_center_[0] += up.x * left_y * pan_speed;
    camera_center_[1] += up.y * left_y * pan_speed;
    camera_center_[2] += up.z * left_y * pan_speed;
  }

  // Back triggers: Forward/backward movement
  // Left trigger = backward, Right trigger = forward
  float move_input = controller_trigger_right_ - controller_trigger_left_;
  if (std::abs(move_input) > 0.1f) {
    glm::mat4 view = GetViewMatrix();
    glm::vec3 forward = -glm::vec3(view[0][2], view[1][2], view[2][2]);  // Negative Z is forward

    float move_speed = controller_move_speed_ * camera_distance_ * delta_time;
    camera_center_[0] += forward.x * move_input * move_speed;
    camera_center_[1] += forward.y * move_input * move_speed;
    camera_center_[2] += forward.z * move_input * move_speed;
  }
}

void Viewer::UpdateCamera() {
  if (scroll_offset_ != 0.0f) {
    camera_distance_ *= (1.0f - scroll_offset_ * 0.1f);
    camera_distance_ = std::max(0.1f, std::min(1000.0f, camera_distance_));
    scroll_offset_ = 0.0f;
  }
}

void Viewer::RecreateSwapchainResources() {
  auto device = renderer_->device();
  VkDevice vk_device = device->device();

  // Wait for all pending operations to complete before destroying resources
  device->WaitIdle();

  // Destroy old ImGui framebuffers BEFORE recreating swapchain
  // (framebuffers depend on swapchain image views, so they must be destroyed first)
  if (!imgui_framebuffers_.empty()) {
    for (auto framebuffer : imgui_framebuffers_) {
      if (framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(vk_device, framebuffer, nullptr);
      }
    }
    imgui_framebuffers_.clear();
  }

  // Destroy old render pass if it exists
  if (imgui_render_pass_ != VK_NULL_HANDLE) {
    vkDestroyRenderPass(vk_device, imgui_render_pass_, nullptr);
    imgui_render_pass_ = VK_NULL_HANDLE;
  }

  // Destroy old semaphores and fences
  for (auto semaphore : image_acquired_semaphores_) {
    vkDestroySemaphore(vk_device, semaphore, nullptr);
  }
  for (auto semaphore : render_finished_semaphores_) {
    vkDestroySemaphore(vk_device, semaphore, nullptr);
  }
  for (auto fence : render_finished_fences_) {
    vkDestroyFence(vk_device, fence, nullptr);
  }

  if (staging_buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(device->allocator(), staging_buffer_, staging_allocation_);
    staging_buffer_ = VK_NULL_HANDLE;
    staging_allocation_ = VK_NULL_HANDLE;
    staging_size_ = 0;
  }

  // Recreate swapchain (this will invalidate old image views)
  swapchain_->Recreate(width_, height_);

  // Recreate ImGui render pass and framebuffers with new swapchain
  // Note: We don't need to shutdown/reinitialize ImGui Vulkan - only the render pass and framebuffers need to be recreated
  // The descriptor pool and ImGui state persist across swapchain recreations
  CreateImGuiRenderPass(device, swapchain_->format());

  // Update ImGui with the new render pass
  if (gui_) {
    gui_->UpdateRenderPass(imgui_render_pass_);
  }

  // Create new semaphores and fences
  uint32_t image_count = swapchain_->image_count();
  image_acquired_semaphores_.resize(image_count);
  render_finished_semaphores_.resize(image_count);
  render_finished_fences_.resize(image_count);
  semaphore_last_image_.resize(image_count, UINT32_MAX);  // Initialize to invalid index
  image_has_been_presented_.resize(image_count, false);
  command_buffers_.assign(image_count, nullptr);

  VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (uint32_t i = 0; i < image_count; ++i) {
    vkCreateSemaphore(vk_device, &semaphore_info, nullptr, &image_acquired_semaphores_[i]);
    vkCreateSemaphore(vk_device, &semaphore_info, nullptr, &render_finished_semaphores_[i]);
    vkCreateFence(vk_device, &fence_info, nullptr, &render_finished_fences_[i]);
  }
}

void Viewer::CreateImGuiRenderPass(std::shared_ptr<gpu::Device> device, VkFormat swapchain_format) {
  VkDevice vk_device = device->device();

  // Create a simple render pass for ImGui (compatible with swapchain format)
  VkAttachmentDescription color_attachment = {};
  color_attachment.format = swapchain_format;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // Load existing content for proper blending
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;  // Image is transitioned to this layout before render pass
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference color_attachment_ref = {};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;

  VkSubpassDependency dependency = {};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  // Accept image from any previous layout (UNDEFINED, PRESENT_SRC_KHR, or COLOR_ATTACHMENT_OPTIMAL)
  // Use TOP_OF_PIPE to accept from any previous stage
  dependency.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  dependency.srcAccessMask = 0;  // No access needed, just waiting for layout transition
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo render_pass_info = {};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &color_attachment;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies = &dependency;

  if (vkCreateRenderPass(vk_device, &render_pass_info, nullptr, &imgui_render_pass_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create ImGui render pass");
  }

  // Create framebuffers for each swapchain image
  uint32_t image_count = swapchain_->image_count();
  imgui_framebuffers_.resize(image_count);

  for (uint32_t i = 0; i < image_count; ++i) {
    VkImageView image_view = swapchain_->image_view(i);

    VkFramebufferCreateInfo framebuffer_info = {};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = imgui_render_pass_;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = &image_view;
    framebuffer_info.width = swapchain_->width();
    framebuffer_info.height = swapchain_->height();
    framebuffer_info.layers = 1;

    if (vkCreateFramebuffer(vk_device, &framebuffer_info, nullptr, &imgui_framebuffers_[i]) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create ImGui framebuffer");
    }
  }
}

void Viewer::RenderFrame() {
  if (!splats_) return;

  // Track frame time
  auto current_time = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_frame_time_);
  current_frame_time_ms_ = elapsed.count() / 1000.0f;
  last_frame_time_ = current_time;

  // Store frame time in history (circular buffer)
  if (frame_times_ms_.size() >= FRAME_HISTORY_SIZE) {
    frame_times_ms_.erase(frame_times_ms_.begin());
  }
  frame_times_ms_.push_back(current_frame_time_ms_);

  // Update window size
  int fb_width, fb_height;
  SDL_GetWindowSize(window_, &fb_width, &fb_height);
  if (fb_width != static_cast<int>(width_) || fb_height != static_cast<int>(height_)) {
    width_ = fb_width;
    height_ = fb_height;
    renderer_->device()->WaitIdle();
    RecreateSwapchainResources();
  }

  // Check if swapchain needs recreation
  if (swapchain_->ShouldRecreate()) {
    renderer_->device()->WaitIdle();
    RecreateSwapchainResources();
  }

  auto device = renderer_->device();
  VkDevice vk_device = device->device();

  // Acquire swapchain image using binary semaphore
  uint32_t image_index;
  uint32_t semaphore_index = frame_counter_ % swapchain_->image_count();
  VkSemaphore acquire_semaphore = image_acquired_semaphores_[semaphore_index];

  // Wait for fence of image that was last using this semaphore (if any)
  if (semaphore_last_image_[semaphore_index] != UINT32_MAX) {
    VkFence last_fence = render_finished_fences_[semaphore_last_image_[semaphore_index]];
    vkWaitForFences(vk_device, 1, &last_fence, VK_TRUE, UINT64_MAX);
  }

  VkResult acquire_result = vkAcquireNextImageKHR(vk_device, swapchain_->handle(), UINT64_MAX, acquire_semaphore, VK_NULL_HANDLE, &image_index);
  if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
      // Swapchain needs recreation, will be handled next frame
      return;
    }
    return;
  }

  // Wait for fence from previous frame using this image
  VkFence render_fence = render_finished_fences_[image_index];
  vkWaitForFences(vk_device, 1, &render_fence, VK_TRUE, UINT64_MAX);
  vkResetFences(vk_device, 1, &render_fence);

  // Get view and projection matrices
  glm::mat4 view = GetViewMatrix();
  float aspect = static_cast<float>(width_) / static_cast<float>(height_);
  glm::mat4 projection = glm::perspective(glm::radians(camera_fov_), aspect, camera_near_, camera_far_);

  // Render to buffer
  core::DrawOptions draw_options;
  draw_options.view = view;
  draw_options.projection = projection;
  draw_options.width = width_;
  draw_options.height = height_;
  draw_options.background = glm::vec3(0.1f, 0.1f, 0.1f);
  draw_options.eps2d = 0.3f;
  draw_options.sh_degree = -1;
  draw_options.visualize_depth = visualize_depth_;
  // If auto-range button was clicked, enable it for this frame only
  bool compute_auto_range = depth_auto_range_;
  draw_options.depth_auto_range = compute_auto_range;
  // Depth range in meters (no normalization needed - shader uses meters directly)
  draw_options.depth_z_min = depth_z_min_;
  draw_options.depth_z_max = depth_z_max_;
  draw_options.camera_near = camera_near_;
  draw_options.camera_far = camera_far_;
  // Set output pointers for auto-range (one-time computation)
  if (compute_auto_range) {
    draw_options.depth_z_min_out = &depth_z_min_;
    draw_options.depth_z_max_out = &depth_z_max_;
    // Reset the flag after setting it (one-time computation)
    depth_auto_range_ = false;
  }

  size_t image_size = static_cast<size_t>(width_) * static_cast<size_t>(height_) * 4;
  if (image_size == 0) {
    return;
  }

  if (image_data_.size() != image_size) {
    image_data_.resize(image_size);
  }

  auto rendered_image = renderer_->Draw(splats_, draw_options, image_data_.data());
  rendered_image->Wait();

  // Copy buffer to swapchain image (3D scene)
  auto gq = device->graphics_queue();
  auto& command = command_buffers_[image_index];
  if (!command) {
    command = gq->AllocateCommandBuffer();
  }
  VkCommandBuffer command_buffer = *command;

  VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkResetCommandBuffer(command_buffer, 0);
  vkBeginCommandBuffer(command_buffer, &begin_info);

  // Create staging buffer (if needed) and copy image data
  if (staging_buffer_ == VK_NULL_HANDLE || staging_size_ != image_size) {
    if (staging_buffer_ != VK_NULL_HANDLE) {
      vmaDestroyBuffer(device->allocator(), staging_buffer_, staging_allocation_);
    }

    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = image_size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    vmaCreateBuffer(device->allocator(), &buffer_info, &alloc_info, &staging_buffer_, &staging_allocation_, nullptr);
    staging_size_ = image_size;
  }

  void* mapped;
  vmaMapMemory(device->allocator(), staging_allocation_, &mapped);
  std::memcpy(mapped, image_data_.data(), image_size);
  vmaFlushAllocation(device->allocator(), staging_allocation_, 0, image_size);
  vmaUnmapMemory(device->allocator(), staging_allocation_);

  // Transition swapchain image to transfer dst
  // After vkAcquireNextImageKHR, the image layout is undefined on first use
  // After presentation, the image is in PRESENT_SRC_KHR layout
  // We need to wait for the acquire semaphore before accessing the image
  VkImage swapchain_image = swapchain_->image(image_index);

  VkImageMemoryBarrier2 barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  if (image_has_been_presented_[image_index]) {
    // Image was previously presented, transition from PRESENT_SRC_KHR
    // The acquire semaphore ensures the image is ready before we transition
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    barrier.srcAccessMask = 0;  // No access needed, just waiting for acquire
    barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  } else {
    // First use of this image, transition from UNDEFINED
    // The acquire semaphore ensures the image is ready before we transition
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;  // Start from top of pipe
    barrier.srcAccessMask = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  }
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.image = swapchain_image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

  VkDependencyInfo dep_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep_info.imageMemoryBarrierCount = 1;
  dep_info.pImageMemoryBarriers = &barrier;
  // The barrier will wait for the acquire semaphore (via wait_info in submit)
  // Using TOP_OF_PIPE ensures we wait for the acquire before any operations
  vkCmdPipelineBarrier2(command_buffer, &dep_info);

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
  region.imageExtent = {width_, height_, 1};
  vkCmdCopyBufferToImage(command_buffer, staging_buffer_, swapchain_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  // Transition swapchain image to color attachment for ImGui rendering
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  vkCmdPipelineBarrier2(command_buffer, &dep_info);

  // Render ImGui GUI on top of 3D scene using Vulkan
  if (gui_ && !showing_title_screen_) {
    VkFramebuffer framebuffer = imgui_framebuffers_[image_index];
    gui_->RenderAllPanels(command_buffer, framebuffer, width_, height_,
                          showing_title_screen_, stats_panel_open_, visual_panel_open_,
                          frame_times_ms_, current_frame_time_ms_, visualize_depth_,
                          depth_auto_range_, depth_z_min_, depth_z_max_);
  }

  // Transition swapchain image to present
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
  barrier.dstAccessMask = 0;
  barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  vkCmdPipelineBarrier2(command_buffer, &dep_info);

  vkEndCommandBuffer(command_buffer);

  // Submit command buffer with binary semaphores
  VkSemaphore render_finished_semaphore = render_finished_semaphores_[image_index];

  VkSemaphoreSubmitInfo wait_info = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  wait_info.semaphore = acquire_semaphore;
  // Wait at top of pipe to ensure image is ready before any layout transitions
  // This ensures the acquire semaphore is waited on before any image operations
  wait_info.stageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;

  VkCommandBufferSubmitInfo cmd_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cmd_info.commandBuffer = command_buffer;

  VkSemaphoreSubmitInfo signal_info = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  signal_info.semaphore = render_finished_semaphore;
  signal_info.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

  VkSubmitInfo2 submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit_info.waitSemaphoreInfoCount = 1;
  submit_info.pWaitSemaphoreInfos = &wait_info;
  submit_info.commandBufferInfoCount = 1;
  submit_info.pCommandBufferInfos = &cmd_info;
  submit_info.signalSemaphoreInfoCount = 1;
  submit_info.pSignalSemaphoreInfos = &signal_info;

  VkQueue graphics_queue_handle = *gq;
  vkQueueSubmit2(graphics_queue_handle, 1, &submit_info, render_fence);

  // Present with binary semaphore
  swapchain_->Present(graphics_queue_handle, image_index, render_finished_semaphore);

  // Mark this image as having been presented
  image_has_been_presented_[image_index] = true;

  // Track which image is using this semaphore
  semaphore_last_image_[semaphore_index] = image_index;

  frame_counter_++;
}

void Viewer::WarmUpFilePicker() {
#ifdef __APPLE__
  // Pre-initialize the file picker system by creating and configuring a panel
  // This warms up macOS's file dialog system so it opens instantly when needed
  void (^warmup_block)(void) = ^{
    @autoreleasepool {
      // Create a panel with the same configuration we'll use later
      NSOpenPanel* panel = [NSOpenPanel openPanel];

      // CRITICAL: Set directory FIRST before any other configuration
      NSArray* paths = NSSearchPathForDirectoriesInDomains(NSDesktopDirectory, NSUserDomainMask, YES);
      if ([paths count] > 0) {
        NSString* desktopPath = [paths objectAtIndex:0];
        NSURL* desktopURL = [NSURL fileURLWithPath:desktopPath];
        [panel setDirectoryURL:desktopURL];
      }

      // Configure with same optimizations as ShowFilePicker
      [panel setTitle:@"Open PLY File"];
      #pragma clang diagnostic push
      #pragma clang diagnostic ignored "-Wdeprecated-declarations"
      [panel setAllowedFileTypes:@[@"ply"]];
      #pragma clang diagnostic pop
      [panel setAllowsMultipleSelection:NO];
      [panel setCanChooseDirectories:NO];
      [panel setCanChooseFiles:YES];
      [panel setCanCreateDirectories:NO];
      [panel setResolvesAliases:NO];
      [panel setShowsHiddenFiles:NO];
      [panel setCanSelectHiddenExtension:NO];
      [panel setTreatsFilePackagesAsDirectories:NO];
      [panel setAccessoryView:nil];

      if ([panel respondsToSelector:@selector(setAccessoryViewDisclosed:)]) {
        [panel setAccessoryViewDisclosed:NO];
      }

      // Just creating and configuring the panel is enough to warm up the system
      // We don't need to show it - the configuration alone initializes the dialog system
    }
  };

  // Execute warm-up on main thread (async to not block initialization)
  if ([NSThread isMainThread]) {
    warmup_block();
  } else {
    dispatch_async(dispatch_get_main_queue(), warmup_block);
  }
#endif
}

std::string Viewer::ShowFilePicker() {
#ifdef __APPLE__
  // Get the Cocoa window from SDL3 using properties
  SDL_PropertiesID props = SDL_GetWindowProperties(window_);
  NSWindow* ns_window = static_cast<NSWindow*>(
    SDL_GetPointerProperty(props, SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr));
  if (!ns_window) {
    return "";
  }

  // Ensure we're on the main thread for UI operations
  __block std::string result = "";
  __block bool completed = false;

  void (^show_picker_block)(void) = ^{
    @autoreleasepool {
      // Create a fresh panel each time to avoid state issues
      NSOpenPanel* panel = [NSOpenPanel openPanel];

      // CRITICAL: Set directory FIRST before any other configuration
      // This prevents macOS from scanning the entire filesystem
      NSArray* paths = NSSearchPathForDirectoriesInDomains(NSDesktopDirectory, NSUserDomainMask, YES);
      if ([paths count] > 0) {
        NSString* desktopPath = [paths objectAtIndex:0];
        NSURL* desktopURL = [NSURL fileURLWithPath:desktopPath];
        [panel setDirectoryURL:desktopURL];
      }

      // Optimize panel configuration for fastest possible opening
      [panel setTitle:@"Open PLY File"];
      #pragma clang diagnostic push
      #pragma clang diagnostic ignored "-Wdeprecated-declarations"
      [panel setAllowedFileTypes:@[@"ply"]];
      #pragma clang diagnostic pop
      [panel setAllowsMultipleSelection:NO];
      [panel setCanChooseDirectories:NO];
      [panel setCanChooseFiles:YES];

      // Disable ALL features that might slow down initialization
      [panel setCanCreateDirectories:NO];
      [panel setResolvesAliases:NO];  // Disable alias resolution for speed
      [panel setShowsHiddenFiles:NO];
      [panel setCanSelectHiddenExtension:NO];
      [panel setTreatsFilePackagesAsDirectories:NO];

      // CRITICAL: Disable preview panel - this is a major source of delay
      [panel setAccessoryView:nil];

      // Disable any accessory views that might cause delays
      if ([panel respondsToSelector:@selector(setAccessoryViewDisclosed:)]) {
        [panel setAccessoryViewDisclosed:NO];
      }

      // Try using runModal for faster response when on main thread
      // This is faster than beginSheetModalForWindow but requires event processing
      if ([NSThread isMainThread]) {
        // Use runModal which is typically faster than sheet modal
        NSModalResponse response = [panel runModal];
        if (response == NSModalResponseOK) {
          NSURL* url = [[panel URLs] firstObject];
          if (url) {
            const char* path = [[url path] UTF8String];
            result = std::string(path);
          }
        }
        completed = true;
      } else {
        // For non-main thread, use sheet modal with completion handler
        [panel beginSheetModalForWindow:ns_window completionHandler:^(NSModalResponse response) {
          if (response == NSModalResponseOK) {
            NSURL* url = [[panel URLs] firstObject];
            if (url) {
              const char* path = [[url path] UTF8String];
              result = std::string(path);
            }
          }
          completed = true;
        }];
      }
    }
  };

  // Execute panel creation on main thread
  if ([NSThread isMainThread]) {
    show_picker_block();
  } else {
    dispatch_sync(dispatch_get_main_queue(), show_picker_block);
  }

  // Process events until the sheet completes (only needed for non-main thread case)
  if (!completed && !should_close_) {
    NSRunLoop* run_loop = [NSRunLoop currentRunLoop];

    while (!completed && !should_close_) {
      // Use a very short timeout for immediate responsiveness
      NSDate* deadline = [NSDate dateWithTimeIntervalSinceNow:0.001];

      // Process Cocoa events using the run loop
      // This processes events for the sheet modal dialog
      [run_loop runMode:NSDefaultRunLoopMode beforeDate:deadline];

      // Process SDL3 events in a tight loop
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        // Process all pending SDL events
      }
    }
  }

  return result;
#else
  // For non-macOS platforms, return empty string (could implement with other file pickers)
  return "";
#endif
}

void Viewer::RenderTitleScreen() {
  // Update window size
  int fb_width, fb_height;
  SDL_GetWindowSize(window_, &fb_width, &fb_height);
  if (fb_width != static_cast<int>(width_) || fb_height != static_cast<int>(height_)) {
    width_ = fb_width;
    height_ = fb_height;
    renderer_->device()->WaitIdle();
    RecreateSwapchainResources();
  }

  // Check if swapchain needs recreation
  if (swapchain_->ShouldRecreate()) {
    renderer_->device()->WaitIdle();
    RecreateSwapchainResources();
  }

  auto device = renderer_->device();
  VkDevice vk_device = device->device();

  // Acquire swapchain image
  uint32_t image_index;
  uint32_t semaphore_index = frame_counter_ % swapchain_->image_count();
  VkSemaphore acquire_semaphore = image_acquired_semaphores_[semaphore_index];

  // Wait for fence of image that was last using this semaphore (if any)
  if (semaphore_last_image_[semaphore_index] != UINT32_MAX) {
    VkFence last_fence = render_finished_fences_[semaphore_last_image_[semaphore_index]];
    vkWaitForFences(vk_device, 1, &last_fence, VK_TRUE, UINT64_MAX);
  }

  VkResult acquire_result = vkAcquireNextImageKHR(vk_device, swapchain_->handle(), UINT64_MAX, acquire_semaphore, VK_NULL_HANDLE, &image_index);
  if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
      return;
    }
    return;
  }

  // Wait for fence
  VkFence render_fence = render_finished_fences_[image_index];
  vkWaitForFences(vk_device, 1, &render_fence, VK_TRUE, UINT64_MAX);
  vkResetFences(vk_device, 1, &render_fence);

  // Get command buffer for rendering
  auto gq = device->graphics_queue();
  auto& command = command_buffers_[image_index];
  if (!command) {
    command = gq->AllocateCommandBuffer();
  }
  VkCommandBuffer command_buffer = *command;

  VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkResetCommandBuffer(command_buffer, 0);
  vkBeginCommandBuffer(command_buffer, &begin_info);

  // Fill background with dark gray for title screen
  VkImage swapchain_image = swapchain_->image(image_index);
  VkImageView swapchain_image_view = swapchain_->image_view(image_index);

  // Transition swapchain image to color attachment
  VkImageMemoryBarrier2 barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  if (image_has_been_presented_[image_index]) {
    // Image was previously presented, transition from PRESENT_SRC_KHR
    // The acquire semaphore ensures the image is ready before we transition
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    barrier.srcAccessMask = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  } else {
    // First use of this image, transition from UNDEFINED
    // The acquire semaphore ensures the image is ready before we transition
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;  // Start from top of pipe
    barrier.srcAccessMask = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  }
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  barrier.image = swapchain_image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

  VkDependencyInfo dep_info = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep_info.imageMemoryBarrierCount = 1;
  dep_info.pImageMemoryBarriers = &barrier;
  // The barrier will wait for the acquire semaphore (via wait_info in submit)
  // Using TOP_OF_PIPE ensures we wait for the acquire before any operations
  vkCmdPipelineBarrier2(command_buffer, &dep_info);

  // Render title screen using Vulkan
  if (gui_) {
    auto show_picker = [this]() -> std::string { return ShowFilePicker(); };
    VkFramebuffer framebuffer = imgui_framebuffers_[image_index];
    gui_->RenderTitleScreen(command_buffer, framebuffer, width_, height_,
                            showing_title_screen_, pending_ply_path_, show_picker);
  }

  // Transition swapchain image to present
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
  barrier.dstAccessMask = 0;
  barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  vkCmdPipelineBarrier2(command_buffer, &dep_info);

  vkEndCommandBuffer(command_buffer);

  // Submit command buffer
  VkSemaphore render_finished_semaphore = render_finished_semaphores_[image_index];

  VkSemaphoreSubmitInfo wait_info = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  wait_info.semaphore = acquire_semaphore;
  // Wait at top of pipe to ensure image is ready before any layout transitions
  wait_info.stageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;

  VkCommandBufferSubmitInfo cmd_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cmd_info.commandBuffer = command_buffer;

  VkSemaphoreSubmitInfo signal_info = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  signal_info.semaphore = render_finished_semaphore;
  signal_info.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

  VkSubmitInfo2 submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit_info.waitSemaphoreInfoCount = 1;
  submit_info.pWaitSemaphoreInfos = &wait_info;
  submit_info.commandBufferInfoCount = 1;
  submit_info.pCommandBufferInfos = &cmd_info;
  submit_info.signalSemaphoreInfoCount = 1;
  submit_info.pSignalSemaphoreInfos = &signal_info;

  VkQueue graphics_queue_handle = *gq;
  vkQueueSubmit2(graphics_queue_handle, 1, &submit_info, render_fence);

  // Present
  swapchain_->Present(graphics_queue_handle, image_index, render_finished_semaphore);

  // Track which image is using this semaphore (for title screen)
  semaphore_last_image_[semaphore_index] = image_index;

  image_has_been_presented_[image_index] = true;
  frame_counter_++;
}

void Viewer::Run(const std::string& ply_path) {
  // If a PLY path is provided, load it immediately
  if (!ply_path.empty()) {
    pending_ply_path_ = ply_path;
    showing_title_screen_ = false;
  }

  while (!should_close_) {
    // Process SDL3 events
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      // Let ImGui process events first
      if (gui_) {
        ImGui_ImplSDL3_ProcessEvent(&event);
      }

      if (event.type == SDL_EVENT_QUIT) {
        should_close_ = true;
      } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN || event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
        // Check if ImGui wants to capture mouse input
        bool imgui_wants_mouse = false;
        if (gui_) {
          ImGuiIO& io = ImGui::GetIO();
          imgui_wants_mouse = io.WantCaptureMouse;
        }

        // Only process camera controls if ImGui doesn't want the mouse
        if (!imgui_wants_mouse) {
          const SDL_MouseButtonEvent& btn = event.button;
          if (btn.button == SDL_BUTTON_LEFT) {
            mouse_left_pressed_ = btn.down;
            if (btn.down) {
              // Start arcball: project initial click position to sphere
              current_mouse_x_ = btn.x;
              current_mouse_y_ = btn.y;
              last_mouse_x_ = btn.x;
              last_mouse_y_ = btn.y;

              // Store initial click position on sphere and current rotation as base
              arcball_start_ = ProjectToSphere(static_cast<float>(btn.x), static_cast<float>(btn.y),
                                               static_cast<float>(width_), static_cast<float>(height_));
              arcball_base_rotation_ = camera_rotation_;  // Store current rotation as base
              arcball_active_ = true;
            } else {
              // Button released - arcball drag is complete
              arcball_active_ = false;
            }
          } else if (btn.button == SDL_BUTTON_RIGHT) {
            mouse_right_pressed_ = btn.down;
            if (btn.down) {
              last_mouse_x_ = btn.x;
              last_mouse_y_ = btn.y;
            }
          }
        } else {
          // If ImGui wants the mouse, release any pressed buttons
          mouse_left_pressed_ = false;
          mouse_right_pressed_ = false;
        }
      } else if (event.type == SDL_EVENT_MOUSE_MOTION) {
        // Check if ImGui wants to capture mouse input
        bool imgui_wants_mouse = false;
        if (gui_) {
          ImGuiIO& io = ImGui::GetIO();
          imgui_wants_mouse = io.WantCaptureMouse;
        }

        const SDL_MouseMotionEvent& motion = event.motion;
        current_mouse_x_ = motion.x;
        current_mouse_y_ = motion.y;

        // Only process camera controls if ImGui doesn't want the mouse
        if (!imgui_wants_mouse) {
          if (mouse_left_pressed_ && arcball_active_) {
            // Project current mouse position to sphere
            glm::vec3 arcball_current = ProjectToSphere(static_cast<float>(motion.x), static_cast<float>(motion.y),
                                                        static_cast<float>(width_), static_cast<float>(height_));

            // Compute rotation from initial click position to current position
            // This ensures the arcball resets properly on each new drag
            glm::quat rotation_delta = ComputeArcballRotation(arcball_start_, arcball_current);

            // Apply rotation: new_rotation = base_rotation * rotation_from_start_to_current
            // This gives smooth rotation that resets correctly when starting a new drag
            camera_rotation_ = arcball_base_rotation_ * rotation_delta;
          } else if (mouse_right_pressed_) {
            double dx = motion.x - last_mouse_x_;
            double dy = motion.y - last_mouse_y_;

            glm::mat4 view = GetViewMatrix();
            glm::vec3 right = glm::vec3(view[0][0], view[1][0], view[2][0]);
            glm::vec3 up = glm::vec3(view[0][1], view[1][1], view[2][1]);

            float pan_speed = camera_distance_ * 0.001f;
            camera_center_[0] -= right.x * static_cast<float>(dx) * pan_speed;
            camera_center_[1] -= right.y * static_cast<float>(dx) * pan_speed;
            camera_center_[2] -= right.z * static_cast<float>(dx) * pan_speed;
            camera_center_[0] -= up.x * static_cast<float>(dy) * pan_speed;
            camera_center_[1] -= up.y * static_cast<float>(dy) * pan_speed;
            camera_center_[2] -= up.z * static_cast<float>(dy) * pan_speed;

            last_mouse_x_ = motion.x;
            last_mouse_y_ = motion.y;
          }
        } else {
          // If ImGui wants the mouse, release any pressed buttons
          mouse_left_pressed_ = false;
          mouse_right_pressed_ = false;
        }
      } else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
        // Check if ImGui wants to capture mouse input
        bool imgui_wants_mouse = false;
        if (gui_) {
          ImGuiIO& io = ImGui::GetIO();
          imgui_wants_mouse = io.WantCaptureMouse;
        }

        // Only process scroll for camera if ImGui doesn't want the mouse
        if (!imgui_wants_mouse) {
          const SDL_MouseWheelEvent& wheel = event.wheel;
          scroll_offset_ += static_cast<float>(wheel.y);
        }
      } else if (event.type == SDL_EVENT_KEY_DOWN) {
        const SDL_KeyboardEvent& key_event = event.key;
        if (key_event.key == SDLK_ESCAPE) {
          if (showing_title_screen_) {
            Close();
          } else {
            showing_title_screen_ = true;
            splats_.reset();
          }
        } else if (key_event.key == SDLK_O) {
          // Handle Cmd+O (macOS) or Ctrl+O (Linux/Windows)
          bool is_mod_pressed = (key_event.mod & SDL_KMOD_GUI) || (key_event.mod & SDL_KMOD_CTRL);
          if (is_mod_pressed) {
            std::string path = ShowFilePicker();
            if (!path.empty()) {
              pending_ply_path_ = path;
            }
          }
        } else if (key_event.key == SDLK_V && !showing_title_screen_) {
          // Toggle visual panel with 'V' key
          visual_panel_open_ = !visual_panel_open_;
        }
      } else if (event.type == SDL_EVENT_GAMEPAD_ADDED) {
        // Controller connected
        const SDL_GamepadDeviceEvent& device_event = event.gdevice;
        if (!controller_ && SDL_IsGamepad(device_event.which)) {
          controller_ = SDL_OpenGamepad(device_event.which);
          if (controller_) {
            std::cout << "Game controller connected: " << SDL_GetGamepadName(controller_) << std::endl;
          }
        }
      } else if (event.type == SDL_EVENT_GAMEPAD_REMOVED) {
        // Controller disconnected
        const SDL_GamepadDeviceEvent& device_event = event.gdevice;
        if (controller_ && SDL_GetGamepadID(controller_) == device_event.which) {
          std::cout << "Game controller disconnected" << std::endl;
          SDL_CloseGamepad(controller_);
          controller_ = nullptr;
        }
      } else if (event.type == SDL_EVENT_GAMEPAD_AXIS_MOTION && controller_) {
        // Controller axis motion
        const SDL_GamepadAxisEvent& axis_event = event.gaxis;
        float value = static_cast<float>(axis_event.value) / 32767.0f;  // Normalize to [-1, 1]

        if (axis_event.axis == SDL_GAMEPAD_AXIS_LEFTX) {
          controller_left_stick_x_ = value;
        } else if (axis_event.axis == SDL_GAMEPAD_AXIS_LEFTY) {
          controller_left_stick_y_ = value;
        } else if (axis_event.axis == SDL_GAMEPAD_AXIS_RIGHTX) {
          controller_right_stick_x_ = value;
        } else if (axis_event.axis == SDL_GAMEPAD_AXIS_RIGHTY) {
          controller_right_stick_y_ = value;
        } else if (axis_event.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER) {
          controller_trigger_left_ = value;
        } else if (axis_event.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
          controller_trigger_right_ = value;
        }
      }
    }

    // Process controller input (if connected and not on title screen)
    if (controller_ && !showing_title_screen_) {
      ProcessControllerInput();
    }

    // Check if a PLY file was selected
    if (!pending_ply_path_.empty()) {
      std::string path_to_load = pending_ply_path_;
      pending_ply_path_.clear();

      std::cout << "Loading PLY file: " << path_to_load << std::endl;

      if (auto bounds_opt = ComputeBoundsFromPly(path_to_load)) {
        const auto& bounds = *bounds_opt;
        if (bounds.valid) {
          glm::vec3 center = (bounds.min + bounds.max) * 0.5f;
          std::cout << "Scene bounds min: (" << bounds.min.x << ", " << bounds.min.y << ", " << bounds.min.z
                    << ") max: (" << bounds.max.x << ", " << bounds.max.y << ", " << bounds.max.z << ")" << std::endl;
          camera_center_[0] = center.x;
          camera_center_[1] = center.y;
          camera_center_[2] = center.z;

          glm::vec3 extents = bounds.max - bounds.min;
          float diagonal = glm::length(extents);
          std::cout << "Scene diagonal length: " << diagonal << std::endl;
          if (!std::isfinite(diagonal) || diagonal <= 0.0f) diagonal = 10.0f;

          camera_distance_ = std::max(5.0f, diagonal * 1.2f);
          camera_near_ = std::max(0.01f, diagonal * 0.01f);
          camera_far_ = std::max(camera_distance_ * 4.0f, diagonal * 4.0f);
          camera_fov_ = 45.0f;
          // Reset camera rotation to default view
          camera_rotation_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);  // Identity quaternion
        }
      }

      splats_ = renderer_->LoadFromPly(path_to_load);
      std::cout << "Loaded " << splats_->size() << " gaussians" << std::endl;
      showing_title_screen_ = false;
    }

    if (showing_title_screen_) {
      RenderTitleScreen();
    } else {
      UpdateCamera();
      RenderFrame();
    }
  }

  renderer_->device()->WaitIdle();
}

void Viewer::Close() {
  should_close_ = true;
}

}  // namespace viewer
}  // namespace vkgs

