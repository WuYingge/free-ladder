#!/bin/bash
# push_to_brain.sh — 只同步 const + etf_data 两个数据目录到脑内助手服务器
# 用法:
#   ./push_to_brain.sh            # 预览 + 确认
#   ./push_to_brain.sh -y         # 自动同步，不确认
#   ./push_to_brain.sh -n         # 只预览，不推送

set -e

REMOTE_HOST="test-claw"
REMOTE_BASE="/root/.openclaw/workspace/opengouzi/incoming"
LOCAL_BASE="/home/gouzi/projects/invest/data"

# 只同步这两个目录（各自由 LOCAL_BASE/<name>/ → REMOTE_BASE/<name>/）
DIRS=("const" "etf_data")

EXCLUDES=(--exclude '.git' --exclude '__pycache__' --exclude '*.pyc')

AUTO=0
DRY=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes)  AUTO=1 ;;
        -n|--dry)  DRY=1 ;;
        -h|--help) echo "用法: $0 [-y] [-n]"; exit 0 ;;
    esac
done

echo "=== 预览变更 ==="
for d in "${DIRS[@]}"; do
    echo "--- $d ---"
    rsync -av --delete --dry-run "${EXCLUDES[@]}" \
        "$LOCAL_BASE/$d/" "$REMOTE_HOST:$REMOTE_BASE/$d/"
done

if [ "$DRY" = "1" ]; then
    echo "（dry-run 模式，未推送）"
    exit 0
fi

if [ "$AUTO" != "1" ]; then
    read -p "确认推送？(y/N) " confirm
    if [ "$confirm" != "y" ]; then
        echo "已取消"
        exit 0
    fi
fi

for d in "${DIRS[@]}"; do
    echo "--- 同步 $d ---"
    rsync -av --delete --progress "${EXCLUDES[@]}" \
        "$LOCAL_BASE/$d/" "$REMOTE_HOST:$REMOTE_BASE/$d/"
done
echo "Done."