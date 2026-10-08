#!/bin/bash
# publish_from_mac.sh を launchd に登録する。
#   平日 7:30 / 8:30（予想）・16:30（採点）、ログイン時にも 1 回
# 使い方: bash scripts/install_mac_schedule.sh
# 解除:   launchctl unload ~/Library/LaunchAgents/com.shigeno.jp-us-leadlag.plist
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.shigeno.jp-us-leadlag"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
case "$REPO" in
  "$HOME/Documents"*|"$HOME/Desktop"*|"$HOME/Downloads"*)
    echo "launchd からは ~/Documents・~/Desktop・~/Downloads の中を読めません（macOS の保護）。" >&2
    echo "ホーム直下に clone して、そこから実行してください: git clone <URL> ~/jp-us-leadlag" >&2
    exit 1;;
esac
mkdir -p "$HOME/Library/LaunchAgents" "$REPO/logs"
ENTRIES=""
for wd in 1 2 3 4 5; do
  for hm in "7 30" "8 30" "16 30"; do
    set -- $hm
    ENTRIES="$ENTRIES<dict><key>Weekday</key><integer>$wd</integer><key>Hour</key><integer>$1</integer><key>Minute</key><integer>$2</integer></dict>"
  done
done
cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array><string>/usr/bin/caffeinate</string><string>-i</string><string>/bin/bash</string><string>$REPO/scripts/publish_from_mac.sh</string></array>
    <key>StartCalendarInterval</key><array>$ENTRIES</array>
    <key>RunAtLoad</key><true/>
    <key>ProcessType</key><string>Standard</string>
    <key>WorkingDirectory</key><string>$REPO</string>
    <key>StandardOutPath</key><string>$REPO/logs/launchd.out</string>
    <key>StandardErrorPath</key><string>$REPO/logs/launchd.err</string>
</dict>
</plist>
PLISTEOF
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "登録しました: $PLIST"
