#include "vkgs/viewer/gui/title_screen.h"

#include "imgui.h"
#include "imgui_impl_sdl3.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace vkgs {
namespace viewer {
namespace gui {

TitleScreen::TitleScreen(const std::string& assets_path) : assets_path_(assets_path) {
  // Load Ubuntu Mono font
  if (!assets_path_.empty()) {
    std::string font_path = assets_path_ + "/Ubuntu_Mono/UbuntuMono-Regular.ttf";
    // Font will be loaded in Initialize() when ImGui context is ready
    font_path_ = font_path;
  }
}

TitleScreen::~TitleScreen() = default;

void TitleScreen::Initialize(SDL_Window* window) {
  window_ = window;

  // Load Ubuntu Mono font at different sizes for different UI elements
  ImGuiIO& io = ImGui::GetIO();

  // Load font at smaller sizes
  if (!font_path_.empty()) {
    font_36_ = io.Fonts->AddFontFromFileTTF(font_path_.c_str(), 24.0f);
    font_48_ = io.Fonts->AddFontFromFileTTF(font_path_.c_str(), 32.0f);

    if (!font_36_) {
      font_36_ = io.Fonts->AddFontDefault();
    }
    if (!font_48_) {
      font_48_ = io.Fonts->AddFontDefault();
    }
  } else {
    font_36_ = io.Fonts->AddFontDefault();
    font_48_ = io.Fonts->AddFontDefault();
  }

  io.FontDefault = font_36_;
  // Note: With newer ImGui backends, Build() is called automatically by the backend

  // Load SVG images as textures (will be done on first render)
  logo_texture_id_ = 0;
  cmd_texture_id_ = 0;
}

void TitleScreen::RenderUI(std::string& pending_ply_path,
                            std::function<std::string()> show_file_picker,
                            std::function<ImTextureID(const std::string&, int, int)> load_svg_texture) {
  if (!window_) return;

  // Set display size (should be set by caller before NewFrame)
  ImGuiIO& io = ImGui::GetIO();

  // Fill background with #1c261c - RGB(28, 38, 28)
  ImDrawList* bg_draw_list = ImGui::GetBackgroundDrawList();
  bg_draw_list->AddRectFilled(ImVec2(0, 0), io.DisplaySize,
                           IM_COL32(28, 38, 28, 255));

  // Set up fullscreen window with no decorations
  ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
  ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);

  ImGui::Begin("TitleScreen", nullptr,
               ImGuiWindowFlags_NoTitleBar |
               ImGuiWindowFlags_NoResize |
               ImGuiWindowFlags_NoMove |
               ImGuiWindowFlags_NoScrollbar |
               ImGuiWindowFlags_NoBackground |
               ImGuiWindowFlags_NoInputs);  // No inputs - we handle clicks separately for buttons

  // Light gray color for all text (#A7A7A7)
  ImVec4 text_color(167.0f / 255.0f, 167.0f / 255.0f, 167.0f / 255.0f, 1.0f);
  ImU32 text_color_u32 = ImGui::ColorConvertFloat4ToU32(text_color);

  // Copyright text at top
  if (font_36_ && font_36_->IsLoaded()) {
    ImGui::PushFont(font_36_);
  }
  ImGui::SetCursorPosY(30.0f);
  ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize("copyright coolant climate inc.").x) * 0.5f);
  ImGui::TextColored(text_color, "copyright coolant climate inc.");

  ImGui::SetCursorPosY(70.0f);  // 30 + 40 spacing
  ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize("do not distribute").x) * 0.5f);
  ImGui::TextColored(text_color, "do not distribute");
  if (font_36_ && font_36_->IsLoaded()) {
    ImGui::PopFont();
  }

  // Calculate content positions for vertical centering
  float logo_height = 88.0f;
  float subtitle_height = 60.0f;
  float button_height = 50.0f;  // Button height (will be reused for buttons)
  float spacing_between_elements = 25.0f;
  float gap_before_buttons = spacing_between_elements * 4.0f;

  float total_content_height = logo_height + spacing_between_elements + subtitle_height + gap_before_buttons + button_height;
  float content_start_y = (ImGui::GetWindowHeight() - total_content_height) * 0.5f;

  // Logo (SVG) - render as image
  float logo_width = 500.0f;
  float logo_x = (ImGui::GetWindowWidth() - logo_width) * 0.5f;
  float logo_y = content_start_y;

  // Load logo texture if not already loaded
  if (logo_texture_id_ == 0) {
    std::string logo_path = assets_path_ + "/logo.svg";
    if (std::filesystem::exists(logo_path)) {
      logo_texture_id_ = load_svg_texture(logo_path, static_cast<int>(logo_width), static_cast<int>(logo_height));
    }
  }

  // Render logo using ImGui::Image if texture is loaded
  if (logo_texture_id_ != 0) {
    ImGui::SetCursorPos(ImVec2(logo_x, logo_y));
    ImGui::Image(logo_texture_id_, ImVec2(logo_width, logo_height));
  } else {
    // Fallback: render as placeholder rectangle
    ImDrawList* window_draw_list = ImGui::GetWindowDrawList();
    window_draw_list->AddRectFilled(ImVec2(logo_x, logo_y),
                                    ImVec2(logo_x + logo_width, logo_y + logo_height),
                                    text_color_u32, 0.0f);
  }

  // "visual inspector" subtitle
  if (font_48_ && font_48_->IsLoaded()) {
    ImGui::PushFont(font_48_);
  }
  float subtitle_y = logo_y + logo_height + spacing_between_elements;
  ImGui::SetCursorPosY(subtitle_y);
  ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize("visual inspector").x) * 0.5f);
  ImGui::TextColored(text_color, "visual inspector");
  if (font_48_ && font_48_->IsLoaded()) {
    ImGui::PopFont();
  }

  // Buttons - side by side
  float button_width = 200.0f;
  float button_horizontal_spacing = 40.0f;
  float button_y = subtitle_y + subtitle_height + gap_before_buttons;

  float total_buttons_width = button_width * 2.0f + button_horizontal_spacing;
  float buttons_start_x = (ImGui::GetWindowWidth() - total_buttons_width) * 0.5f;
  float open_button_x = buttons_start_x;
  float pull_button_x = buttons_start_x + button_width + button_horizontal_spacing;

  // Draw buttons using ImGui buttons
  // We need to enable inputs for the button area
  ImGui::End();  // End the no-inputs window

  // Command icon - load texture if not already loaded
  float cmd_icon_size = 24.0f;
  if (cmd_texture_id_ == 0) {
    std::string cmd_path = assets_path_ + "/command.svg";
    if (std::filesystem::exists(cmd_path)) {
      cmd_texture_id_ = load_svg_texture(cmd_path, static_cast<int>(cmd_icon_size), static_cast<int>(cmd_icon_size));
    }
  }

  // Create separate windows for buttons to handle input
  ImGui::SetNextWindowPos(ImVec2(open_button_x, button_y - button_height * 0.5f), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(button_width, button_height), ImGuiCond_Always);
  ImGui::Begin("OpenButton", nullptr,
               ImGuiWindowFlags_NoTitleBar |
               ImGuiWindowFlags_NoResize |
               ImGuiWindowFlags_NoMove |
               ImGuiWindowFlags_NoScrollbar |
               ImGuiWindowFlags_NoBackground |
               ImGuiWindowFlags_NoNav);

  // Style button to match design (black with dark gray border)
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.05f, 0.05f, 0.05f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));  // Transparent text for button
  ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(40.0f / 255.0f, 40.0f / 255.0f, 40.0f / 255.0f, 1.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));

  if (font_48_ && font_48_->IsLoaded()) {
    ImGui::PushFont(font_48_);
  }

  // Calculate button content layout
  float text_to_shortcut_gap = 20.0f;
  float cmd_to_key_spacing = 4.0f;
  ImVec2 text_size = ImGui::CalcTextSize("OPEN");
  ImVec2 key_o_size = ImGui::CalcTextSize("O");
  float total_content_width = text_size.x + text_to_shortcut_gap + cmd_icon_size + cmd_to_key_spacing + key_o_size.x;

  // Render the button first
  ImGui::SetCursorPos(ImVec2(0, 0));
  bool button_clicked = ImGui::Button("##OpenBtn", ImVec2(button_width, button_height));

  // Allow drawing on top of the button without blocking clicks
  ImGui::SetNextItemAllowOverlap();

  // Get button rect to draw content on top
  ImVec2 button_min = ImGui::GetItemRectMin();
  ImVec2 button_max = ImGui::GetItemRectMax();
  ImDrawList* draw_list = ImGui::GetWindowDrawList();

  // Calculate centered position for content
  float content_start_x = button_min.x + (button_width - total_content_width) * 0.5f;
  float content_y = button_min.y + (button_height - ImGui::GetTextLineHeight()) * 0.5f;

  // Draw text "OPEN"
  if (font_48_ && font_48_->IsLoaded()) {
    draw_list->AddText(font_48_, 32.0f, ImVec2(content_start_x, content_y), text_color_u32, "OPEN");
  } else {
    draw_list->AddText(ImVec2(content_start_x, content_y), text_color_u32, "OPEN");
  }

  // Draw command icon
  float cmd_x = content_start_x + text_size.x + text_to_shortcut_gap;
  float cmd_y = button_min.y + (button_height - cmd_icon_size) * 0.5f;
  if (cmd_texture_id_ != 0) {
    draw_list->AddImage(cmd_texture_id_,
                        ImVec2(cmd_x, cmd_y),
                        ImVec2(cmd_x + cmd_icon_size, cmd_y + cmd_icon_size));
  } else {
    // Fallback: render as placeholder rectangle
    draw_list->AddRectFilled(ImVec2(cmd_x, cmd_y),
                             ImVec2(cmd_x + cmd_icon_size, cmd_y + cmd_icon_size),
                             text_color_u32);
  }

  // Draw "O" key
  float key_o_x = cmd_x + cmd_icon_size + cmd_to_key_spacing;
  if (font_48_ && font_48_->IsLoaded()) {
    draw_list->AddText(font_48_, 32.0f, ImVec2(key_o_x, content_y), text_color_u32, "O");
  } else {
    draw_list->AddText(ImVec2(key_o_x, content_y), text_color_u32, "O");
  }

  // Handle button click
  if (button_clicked) {
    std::string path = show_file_picker();
    if (!path.empty()) {
      pending_ply_path = path;
    }
  }

  if (font_48_ && font_48_->IsLoaded()) {
    ImGui::PopFont();
  }
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor(5);
  ImGui::End();

  // Pull button
  ImGui::SetNextWindowPos(ImVec2(pull_button_x, button_y - button_height * 0.5f), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(button_width, button_height), ImGuiCond_Always);
  ImGui::Begin("PullButton", nullptr,
               ImGuiWindowFlags_NoTitleBar |
               ImGuiWindowFlags_NoResize |
               ImGuiWindowFlags_NoMove |
               ImGuiWindowFlags_NoScrollbar |
               ImGuiWindowFlags_NoBackground);

  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.05f, 0.05f, 0.05f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));  // Transparent text for button
  ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(40.0f / 255.0f, 40.0f / 255.0f, 40.0f / 255.0f, 1.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));

  if (font_48_ && font_48_->IsLoaded()) {
    ImGui::PushFont(font_48_);
  }

  // Calculate button content layout (reuse same spacing values)
  float pull_text_to_shortcut_gap = 20.0f;
  float pull_cmd_to_key_spacing = 4.0f;
  ImVec2 pull_text_size = ImGui::CalcTextSize("PULL");
  ImVec2 key_p_size = ImGui::CalcTextSize("P");
  float pull_total_content_width = pull_text_size.x + pull_text_to_shortcut_gap + cmd_icon_size + pull_cmd_to_key_spacing + key_p_size.x;

  // Render the button first
  ImGui::SetCursorPos(ImVec2(0, 0));
  ImGui::Button("##PullBtn", ImVec2(button_width, button_height));

  // Get button rect to draw content on top
  ImVec2 pull_button_min = ImGui::GetItemRectMin();
  ImVec2 pull_button_max = ImGui::GetItemRectMax();
  ImDrawList* pull_draw_list = ImGui::GetWindowDrawList();

  // Calculate centered position for content
  float pull_content_start_x = pull_button_min.x + (button_width - pull_total_content_width) * 0.5f;
  float pull_content_y = pull_button_min.y + (button_height - ImGui::GetTextLineHeight()) * 0.5f;

  // Draw text "PULL"
  if (font_48_ && font_48_->IsLoaded()) {
    pull_draw_list->AddText(font_48_, 32.0f, ImVec2(pull_content_start_x, pull_content_y), text_color_u32, "PULL");
  } else {
    pull_draw_list->AddText(ImVec2(pull_content_start_x, pull_content_y), text_color_u32, "PULL");
  }

  // Draw command icon
  float pull_cmd_x = pull_content_start_x + pull_text_size.x + pull_text_to_shortcut_gap;
  float pull_cmd_y = pull_button_min.y + (button_height - cmd_icon_size) * 0.5f;
  if (cmd_texture_id_ != 0) {
    pull_draw_list->AddImage(cmd_texture_id_,
                             ImVec2(pull_cmd_x, pull_cmd_y),
                             ImVec2(pull_cmd_x + cmd_icon_size, pull_cmd_y + cmd_icon_size));
  } else {
    // Fallback: render as placeholder rectangle
    pull_draw_list->AddRectFilled(ImVec2(pull_cmd_x, pull_cmd_y),
                                  ImVec2(pull_cmd_x + cmd_icon_size, pull_cmd_y + cmd_icon_size),
                                  text_color_u32);
  }

  // Draw "P" key
  float key_p_x = pull_cmd_x + cmd_icon_size + pull_cmd_to_key_spacing;
  if (font_48_ && font_48_->IsLoaded()) {
    pull_draw_list->AddText(font_48_, 32.0f, ImVec2(key_p_x, pull_content_y), text_color_u32, "P");
  } else {
    pull_draw_list->AddText(ImVec2(key_p_x, pull_content_y), text_color_u32, "P");
  }

  if (font_48_ && font_48_->IsLoaded()) {
    ImGui::PopFont();
  }
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor(5);
  ImGui::End();

  // Version text at bottom
  if (font_36_ && font_36_->IsLoaded()) {
    ImGui::PushFont(font_36_);
  }
  float version_y = io.DisplaySize.y - 40.0f - 30.0f;
  ImGui::SetNextWindowPos(ImVec2(0, version_y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 50.0f), ImGuiCond_Always);
  ImGui::Begin("Version", nullptr,
               ImGuiWindowFlags_NoTitleBar |
               ImGuiWindowFlags_NoResize |
               ImGuiWindowFlags_NoMove |
               ImGuiWindowFlags_NoScrollbar |
               ImGuiWindowFlags_NoBackground |
               ImGuiWindowFlags_NoInputs);

  ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize("version 0.0.0").x) * 0.5f);
  ImGui::TextColored(text_color, "version 0.0.0");

  if (font_36_ && font_36_->IsLoaded()) {
    ImGui::PopFont();
  }
  ImGui::End();
}

bool TitleScreen::HandleClick(int x, int y, int width, int height,
                               std::string& pending_ply_path,
                               std::function<std::string()> show_file_picker) {
  // ImGui handles clicks internally through InvisibleButton
  // This function is kept for compatibility but clicks are handled in RenderUI()
  return false;
}

}  // namespace gui
}  // namespace viewer
}  // namespace vkgs
