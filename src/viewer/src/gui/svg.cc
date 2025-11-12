#include "vkgs/viewer/gui/svg.h"

#include <iostream>
#include <algorithm>

#define NANOSVG_IMPLEMENTATION
#include "nanosvg.h"
#define NANOSVGRAST_IMPLEMENTATION
#include "nanosvgrast.h"

namespace vkgs {
namespace viewer {
namespace gui {

SVG::SVG() {}

SVG::~SVG() {}

void SVG::RenderImage(std::vector<uint8_t>& image_data, int width, int height,
                      const std::string& svg_path, int x, int y, int svg_width, int svg_height,
                      uint8_t tint_r, uint8_t tint_g, uint8_t tint_b) const {
  if (image_data.empty()) return;

  NSVGimage* image = nsvgParseFromFile(svg_path.c_str(), "px", 96.0f);
  if (!image) {
    std::cerr << "Warning: Could not load SVG from " << svg_path << std::endl;
    return;
  }

  // Rasterize SVG to RGBA image
  int raster_width = svg_width;
  int raster_height = svg_height;
  unsigned char* raster = new unsigned char[raster_width * raster_height * 4];

  NSVGrasterizer* rast = nsvgCreateRasterizer();
  nsvgRasterize(rast, image, 0, 0, std::min(static_cast<float>(raster_width) / image->width, static_cast<float>(raster_height) / image->height), raster, raster_width, raster_height, raster_width * 4);
  nsvgDeleteRasterizer(rast);

  // Copy rasterized image to image_data_ with color tinting
  for (int sy = 0; sy < raster_height && (y + sy) < height; ++sy) {
    for (int sx = 0; sx < raster_width && (x + sx) < width; ++sx) {
      int src_idx = (sy * raster_width + sx) * 4;
      int dst_idx = ((y + sy) * width + (x + sx)) * 4;

      if (dst_idx + 3 < static_cast<int>(image_data.size())) {
        unsigned char r = raster[src_idx];
        unsigned char g = raster[src_idx + 1];
        unsigned char b = raster[src_idx + 2];
        unsigned char a = raster[src_idx + 3];

        if (a > 0) {
          // Apply tinting
          float tint_factor = a / 255.0f;
          image_data[dst_idx] = static_cast<uint8_t>(b * tint_factor * (tint_b / 255.0f));
          image_data[dst_idx + 1] = static_cast<uint8_t>(g * tint_factor * (tint_g / 255.0f));
          image_data[dst_idx + 2] = static_cast<uint8_t>(r * tint_factor * (tint_r / 255.0f));
          image_data[dst_idx + 3] = a;
        }
      }
    }
  }

  delete[] raster;
  nsvgDelete(image);
}

}  // namespace gui
}  // namespace viewer
}  // namespace vkgs

