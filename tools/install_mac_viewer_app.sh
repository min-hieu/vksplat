#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIEWER_EXE="$ROOT_DIR/bin/inspector"

if [[ ! -x "$VIEWER_EXE" ]]; then
  echo "Error: $VIEWER_EXE not found or not executable. Build inspector first." >&2
  exit 1
fi

APP_NAME="SplatStreamViewer"
APP_DIR="$HOME/Applications/$APP_NAME.app"

echo "Creating macOS app bundle at $APP_DIR"

mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

cat > "$APP_DIR/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>SplatStreamViewer</string>
  <key>CFBundleIdentifier</key>
  <string>com.splatstream.viewer</string>
  <key>CFBundleName</key>
  <string>SplatStream Viewer</string>
  <key>CFBundleVersion</key>
  <string>1.0</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>LSMinimumSystemVersion</key>
  <string>10.13</string>
  <key>LSApplicationCategoryType</key>
  <string>public.app-category.graphics-design</string>
  <key>LSBackgroundOnly</key>
  <false/>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key>
      <string>PLY Document</string>
      <key>CFBundleTypeRole</key>
      <string>Viewer</string>
      <key>LSHandlerRank</key>
      <string>Alternate</string>
      <key>CFBundleTypeExtensions</key>
      <array>
        <string>ply</string>
      </array>
      <key>CFBundleTypeMIMETypes</key>
      <array>
        <string>application/octet-stream</string>
      </array>
    </dict>
  </array>
</dict>
</plist>
EOF

cat > "$APP_DIR/Contents/MacOS/SplatStreamViewer" <<EOF
#!/usr/bin/env bash
"$VIEWER_EXE" "\$@"
EOF

chmod +x "$APP_DIR/Contents/MacOS/SplatStreamViewer"

echo "Registering app with Launch Services"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP_DIR" >/dev/null 2>&1 || true

echo "Done. You can now open SplatStream Viewer from ~/Applications."
echo "To make it the default app for .ply files:"
echo "  1. In Finder, right-click a .ply file -> Get Info."
echo "  2. Under 'Open with', choose 'SplatStream Viewer'."
echo "  3. Click 'Change All…'."

