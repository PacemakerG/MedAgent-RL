#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="$REPO_ROOT/data/raw_sources"

mkdir -p "$RAW_ROOT"

clone_or_update() {
  local name="$1"
  local url="$2"
  local target="$RAW_ROOT/$name"

  if [[ -d "$target/.git" ]]; then
    echo "[update] $name"
    git -C "$target" pull --ff-only
  elif [[ -e "$target" ]]; then
    echo "[skip] $target 已存在，但不是 Git 仓库。请检查或删除后重试。"
  else
    echo "[clone] $name"
    git clone --depth 1 "$url" "$target"
  fi
}

# 你的 GitHub SSH 已经配置成功，因此默认使用 SSH，避免 HTTPS 443 超时。
clone_or_update "IMCS21" "git@github.com:lemuria-wchen/imcs21.git"
clone_or_update "MedDG" "git@github.com:lwgkzl/MedDG.git"

CHIP_DIR="$RAW_ROOT/CHIP-MDCFNPC"
mkdir -p "$CHIP_DIR"

chip_zip=""
for candidate in \
  "$CHIP_DIR/CHIP-MDCFNPC.zip" \
  "$CHIP_DIR/CHIP_MDCFNPC.zip" \
  "$CHIP_DIR/dataset.zip"; do
  if [[ -f "$candidate" ]]; then
    chip_zip="$candidate"
    break
  fi
done

if [[ -n "$chip_zip" ]]; then
  echo "[unzip] CHIP-MDCFNPC: $chip_zip"
  unzip -q -o "$chip_zip" -d "$CHIP_DIR"
else
  cat <<EOF

[manual] CHIP-MDCFNPC 官方下载需要登录天池，无法通过匿名脚本直接下载。

请执行：
1. 浏览器打开：
   https://tianchi.aliyun.com/dataset/95414
2. 下载 CHIP-MDCFNPC.zip。
3. 放到：
   $CHIP_DIR/CHIP-MDCFNPC.zip
4. 再次运行：
   bash self_scripts/download_raw_datasets.sh

EOF
fi

echo ""
echo "原始数据目录：$RAW_ROOT"
echo ""
find "$RAW_ROOT" -maxdepth 2 -type d | sort
