#!/bin/bash
# Mac から予測ページを生成して GitHub Pages に公開する。
#   平日 7:30 / 8:30 JST : 米国引け後の予想を寄付き前に公開
#   平日 16:30 JST       : 日本の引け後に当日の予想を採点
# GitHub Actions の定時実行は数時間遅れて寄付き後になりがちなため、2026-10 からこちらが主。
# 手動実行:  bash scripts/publish_from_mac.sh
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
mkdir -p logs
exec >> logs/publish.log 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M:%S')"

PY="$REPO/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "▶ 仮想環境を作成します"
  /usr/bin/python3 -m venv "$REPO/.venv" || exit 1
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt || exit 1
fi

# GitHub Actions 側の手動実行などで先に進んでいれば取り込む
git pull -q --rebase --autostash origin main || { echo "pull に失敗"; exit 1; }

"$PY" scripts/build_page.py --outdir docs || { echo "ページ生成に失敗"; exit 1; }

git add docs/index.html docs/history.json docs/holdings.json docs/.nojekyll
if git diff --staged --quiet; then
  echo "変更なし"
else
  git commit -q -m "update: $(date +%Y-%m-%d) の予測（Mac）"
  for i in 1 2 3; do
    git push -q origin main && { echo "公開しました"; break; }
    git pull -q --rebase origin main
    sleep 5
  done
fi
