#include "vkgs/viewer/gui/font.h"

#include <iostream>
#include <fstream>
#include <limits>

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

namespace vkgs {
namespace viewer {
namespace gui {

Font::Font() : font_info_(nullptr) {}

Font::~Font() {
  if (font_info_) {
    delete static_cast<stbtt_fontinfo*>(font_info_);
    font_info_ = nullptr;
  }
}

bool Font::LoadFromFile(const std::string& font_path) {
  std::ifstream font_file(font_path, std::ios::binary | std::ios::ate);
  if (!font_file) {
    std::cerr << "Warning: Could not load font from " << font_path << std::endl;
    return false;
  }

  size_t font_size = font_file.tellg();
  font_file.seekg(0, std::ios::beg);
  font_data_.resize(font_size);
  font_file.read(reinterpret_cast<char*>(font_data_.data()), font_size);

  // Initialize stb_truetype font info
  font_info_ = new stbtt_fontinfo;
  if (!stbtt_InitFont(static_cast<stbtt_fontinfo*>(font_info_), font_data_.data(), 0)) {
    std::cerr << "Warning: Failed to initialize font" << std::endl;
    delete static_cast<stbtt_fontinfo*>(font_info_);
    font_info_ = nullptr;
    return false;
  }

  return true;
}

float Font::MeasureTextWidth(const std::string& text, float size) const {
  if (!font_info_) return 0.0f;

  stbtt_fontinfo* font = static_cast<stbtt_fontinfo*>(font_info_);
  float scale = stbtt_ScaleForPixelHeight(font, size);

  float total_width = 0.0f;
  for (size_t i = 0; i < text.length(); ++i) {
    int char_code = static_cast<unsigned char>(text[i]);
    int advance_width, left_side_bearing;
    stbtt_GetCodepointHMetrics(font, char_code, &advance_width, &left_side_bearing);
    total_width += advance_width * scale;
  }

  return total_width;
}

void Font::RenderText(std::vector<uint8_t>& image_data, int width, int height,
                      const std::string& text, int x, int y, float size,
                      uint8_t r, uint8_t g, uint8_t b) const {
  if (!font_info_ || image_data.empty()) return;

  stbtt_fontinfo* font = static_cast<stbtt_fontinfo*>(font_info_);

  float scale = stbtt_ScaleForPixelHeight(font, size);
  int ascent, descent, line_gap;
  stbtt_GetFontVMetrics(font, &ascent, &descent, &line_gap);

  int x_pos = x;
  for (size_t i = 0; i < text.length(); ++i) {
    int char_code = static_cast<unsigned char>(text[i]);

    int advance_width, left_side_bearing;
    stbtt_GetCodepointHMetrics(font, char_code, &advance_width, &left_side_bearing);

    int x0, y0, x1, y1;
    stbtt_GetCodepointBitmapBox(font, char_code, scale, scale, &x0, &y0, &x1, &y1);

    int bitmap_width = x1 - x0;
    int bitmap_height = y1 - y0;

    if (bitmap_width > 0 && bitmap_height > 0) {
      unsigned char* bitmap = new unsigned char[bitmap_width * bitmap_height];
      stbtt_MakeCodepointBitmap(font, bitmap, bitmap_width, bitmap_height, bitmap_width, scale, scale, char_code);

      int draw_x = x_pos + static_cast<int>(left_side_bearing * scale) + x0;
      int draw_y = y + static_cast<int>(ascent * scale) + y0;

      for (int by = 0; by < bitmap_height; ++by) {
        for (int bx = 0; bx < bitmap_width; ++bx) {
          unsigned char alpha = bitmap[by * bitmap_width + bx];
          if (alpha > 0) {
            int px = draw_x + bx;
            int py = draw_y + by;
            if (px >= 0 && px < width && py >= 0 && py < height) {
              // Blend with existing pixel
              size_t idx = (py * width + px) * 4;
              float alpha_f = alpha / 255.0f;
              image_data[idx] = static_cast<uint8_t>(image_data[idx] * (1.0f - alpha_f) + b * alpha_f);
              image_data[idx + 1] = static_cast<uint8_t>(image_data[idx + 1] * (1.0f - alpha_f) + g * alpha_f);
              image_data[idx + 2] = static_cast<uint8_t>(image_data[idx + 2] * (1.0f - alpha_f) + r * alpha_f);
            }
          }
        }
      }

      delete[] bitmap;
    }

    x_pos += static_cast<int>(advance_width * scale);
  }
}

int Font::CalculateBaselineForCentering(char ch, float size, int center_y) const {
  if (!font_info_) return center_y;

  stbtt_fontinfo* font = static_cast<stbtt_fontinfo*>(font_info_);
  float scale = stbtt_ScaleForPixelHeight(font, size);
  int ascent, descent, line_gap;
  stbtt_GetFontVMetrics(font, &ascent, &descent, &line_gap);

  int char_code = static_cast<unsigned char>(ch);
  int x0, y0, x1, y1;
  stbtt_GetCodepointBitmapBox(font, char_code, scale, scale, &x0, &y0, &x1, &y1);

  // Calculate baseline to center glyph at center_y
  float glyph_center_offset = ascent * scale + (y0 + y1) / 2.0f;
  return static_cast<int>(center_y - glyph_center_offset);
}

int Font::CalculateBaselineForCenteringText(const std::string& text, float size, int center_y) const {
  if (!font_info_ || text.empty()) return center_y;

  stbtt_fontinfo* font = static_cast<stbtt_fontinfo*>(font_info_);
  float scale = stbtt_ScaleForPixelHeight(font, size);
  int ascent, descent, line_gap;
  stbtt_GetFontVMetrics(font, &ascent, &descent, &line_gap);

  int min_y0 = std::numeric_limits<int>::max();
  int max_y1 = std::numeric_limits<int>::min();

  for (char ch : text) {
    int char_code = static_cast<unsigned char>(ch);
    int x0, y0, x1, y1;
    stbtt_GetCodepointBitmapBox(font, char_code, scale, scale, &x0, &y0, &x1, &y1);
    min_y0 = std::min(min_y0, y0);
    max_y1 = std::max(max_y1, y1);
  }

  if (min_y0 > max_y1) {
    return center_y;
  }

  float glyph_center_offset = ascent * scale + (min_y0 + max_y1) / 2.0f;
  return static_cast<int>(center_y - glyph_center_offset);
}

}  // namespace gui
}  // namespace viewer
}  // namespace vkgs

