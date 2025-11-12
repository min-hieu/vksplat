#ifndef VKGS_VIEWER_GUI_FONT_H
#define VKGS_VIEWER_GUI_FONT_H

#include <string>
#include <vector>
#include <cstdint>

namespace vkgs {
namespace viewer {
namespace gui {

class Font {
 public:
  Font();
  ~Font();

  bool LoadFromFile(const std::string& font_path);
  float MeasureTextWidth(const std::string& text, float size) const;
  void RenderText(std::vector<uint8_t>& image_data, int width, int height,
                  const std::string& text, int x, int y, float size,
                  uint8_t r, uint8_t g, uint8_t b) const;

  // Calculate baseline Y position to center a character at a given Y coordinate
  int CalculateBaselineForCentering(char ch, float size, int center_y) const;
  int CalculateBaselineForCenteringText(const std::string& text, float size, int center_y) const;

  bool IsLoaded() const { return font_info_ != nullptr; }

 private:
  std::vector<unsigned char> font_data_;
  void* font_info_;  // stbtt_fontinfo* (cast in implementation)
};

}  // namespace gui
}  // namespace viewer
}  // namespace vkgs

#endif  // VKGS_VIEWER_GUI_FONT_H

