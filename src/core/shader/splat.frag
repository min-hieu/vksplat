#version 460 core

layout(push_constant, std430) uniform PushConstants {
  vec4 background_color;
  uint visualize_depth;
  float depth_z_min;
  float depth_z_max;
  float camera_near;
  float camera_far;
};

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 position;

layout(location = 0) out vec4 out_color;

// Jet colormap: maps [0, 1] to RGB colors (blue -> cyan -> green -> yellow -> red)
vec3 jet_colormap(float t) {
  t = clamp(t, 0.0, 1.0);
  float r = t < 0.5 ? 0.0 : (t < 0.75 ? (t - 0.5) * 4.0 : 1.0);
  float g = t < 0.25 ? t * 4.0 : (t < 0.75 ? 1.0 : (1.0 - t) * 4.0);
  float b = t < 0.25 ? 1.0 : (t < 0.5 ? (0.5 - t) * 4.0 : 0.0);
  return vec3(r, g, b);
}

void main() {
  float gaussian_alpha = exp(-0.5f * dot(position, position));
  float alpha = color.a * gaussian_alpha;

  if (visualize_depth != 0u) {
    // Convert NDC depth to view-space depth (distance from camera in meters)
    // For Vulkan perspective projection: view_z = (near * far) / (far - ndc_z * (far - near))
    float ndc_z = gl_FragCoord.z;
    float view_z = (camera_near * camera_far) / (camera_far - ndc_z * (camera_far - camera_near));

    // Normalize depth in meters using z_min and z_max
    float depth_range = depth_z_max - depth_z_min;
    float normalized_depth = depth_range > 0.0001 ? clamp((view_z - depth_z_min) / depth_range, 0.0, 1.0) : 0.0;
    vec3 depth_color = jet_colormap(normalized_depth);
    out_color = vec4(depth_color * alpha, alpha);
  } else {
    out_color = vec4(color.rgb * alpha, alpha);
  }
}
