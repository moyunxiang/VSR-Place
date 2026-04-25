#!/usr/bin/env bash
# Push cached big artifacts from local mac → AutoDL instance.
#
# Usage:
#   bash scripts/push_to_autodl.sh <ssh-alias-or-user@host[:port]>
#
# Example:
#   bash scripts/push_to_autodl.sh root@connect.bjb1.seetacloud.com:43348
#
# What it copies (~180 MB):
#   - local_artifacts/large-v2.ckpt        (75 MB ChipDiffusion checkpoint)
#   - local_artifacts/config.yaml          (1.7 KB)
#   - local_artifacts/ispd2005dp.tar.xz    (108 MB ISPD2005 raw)
#
# Assumes the AutoDL instance has the repo already cloned at
# /root/autodl-tmp/VSR-Place/.
#
# After this, run on AutoDL:
#   cd /root/autodl-tmp/VSR-Place && bash scripts/bootstrap_autodl.sh

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <ssh-alias-or-user@host[:port]>"
    exit 1
fi

REMOTE="$1"
REPO_LOCAL="$(cd "$(dirname "$0")"/.. && pwd)"
REMOTE_REPO=/root/autodl-tmp/VSR-Place

# Detect ":port" syntax and split for scp -P
SCP_OPTS=()
if [[ "$REMOTE" == *:* ]] && [[ "$REMOTE" != *@*:* ]]; then
    # alias:port form (no @)
    HOST="${REMOTE%:*}"
    PORT="${REMOTE##*:}"
    SCP_OPTS=("-P" "$PORT")
    REMOTE="$HOST"
elif [[ "$REMOTE" == *@*:* ]]; then
    PORT="${REMOTE##*:}"
    REMOTE="${REMOTE%:*}"
    SCP_OPTS=("-P" "$PORT")
fi

echo "Remote: $REMOTE  (port: ${PORT:-22})"
echo "Local repo: $REPO_LOCAL"

# Ensure remote dir exists
ssh ${PORT:+-p $PORT} "$REMOTE" "mkdir -p $REMOTE_REPO/local_artifacts"

# Push
for f in large-v2.ckpt config.yaml ispd2005dp.tar.xz; do
    echo "Pushing $f ..."
    scp "${SCP_OPTS[@]}" "$REPO_LOCAL/local_artifacts/$f" \
        "$REMOTE:$REMOTE_REPO/local_artifacts/$f"
done

echo
echo "Done. Now ssh to AutoDL and run:"
echo "  cd $REMOTE_REPO && bash scripts/bootstrap_autodl.sh"
