#!/bin/bash
# Mac から予測ページを生成して GitHub Pages に公開する。
#   平日 7:30 / 8:30 JST : 米国引け後の予想を寄付き前に公開
#   平日 16:30 JST       : 日本の引け後に当日の予想を採点
# GitHub Actions の定時実行は数時間遅れて寄付き後になりがちなため、2026-10 からこちらが主。
# 手動実行:  bash scripts/publish_from_mac.sh
#
# docs/ は生成物なので、git の行単位のマージはしない。毎回「公開中の最新」に合わせてから、
# 手元の history.json を予想日ごとにまとめ直し（scripts/merge_history.py）、ページを作り直す。
# GitHub Actions（予備）と push がぶつかっても、次の試行で自動的にまとまる。
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

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for attempt in 1 2 3; do
  # 前回の失敗で途中になった rebase / merge が残っていれば片づける
  git rebase --abort >/dev/null 2>&1
  git merge --abort >/dev/null 2>&1
  git fetch -q origin main || { echo "fetch に失敗"; sleep 10; continue; }

  [ -f docs/history.json ] && cp docs/history.json "$TMP/local_history.json"
  git reset -q --hard origin/main
  if [ -f "$TMP/local_history.json" ]; then
    "$PY" scripts/merge_history.py "$TMP/local_history.json" docs/history.json
  fi

  "$PY" scripts/build_page.py --outdir docs || { echo "ページ生成に失敗"; exit 1; }

  git add docs/index.html docs/summary.json docs/history.json docs/holdings.json docs/.nojekyll
  if git diff --staged --quiet; then
    echo "変更なし"
    exit 0
  fi
  git commit -q -m "update: $(date +%Y-%m-%d) の予測（Mac）"
  if git push -q origin main; then
    echo "公開しました"
    exit 0
  fi
  echo "push がぶつかったため、やり直します（$attempt 回目）"
  sleep 5
done
echo "公開に失敗しました（3 回試行）"
exit 1
