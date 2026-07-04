#!/bin/bash
set -e

REMOTE="test-claw:/root/.openclaw/workspace/opengouzi/incoming"
LOCAL="$HOME/projects/free-ladder/data"

echo "=== 预览变更 ==="
rsync -av --delete --dry-run \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$LOCAL/" "$REMOTE/"

echo ""
read -p "确认推送？(y/N) " confirm
if [ "$confirm" != "y" ]; then
    echo "已取消"
    exit 0
fi

rsync -av --delete --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$LOCAL/" "$REMOTE/"

echo "Done."
