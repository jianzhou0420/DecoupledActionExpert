#!/usr/bin/env bash
# Setup script for VLAWorkspace
# Initializes submodules and applies compatibility patches to third-party dependencies.
#
# Usage:
#   git clone --recurse-submodules <repo-url>
#   cd vlaworkspace
#   bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/patches"

# 1. Initialize and update submodules
echo "==> Initializing submodules..."
git submodule update --init --recursive

# 2. Apply patches
declare -A PATCH_TARGETS=(
    ["lerobot.patch"]="third_party/lerobot"
    ["libero.patch"]="third_party/libero"
    ["mimicgen_robomimic.patch"]="third_party/mimicgen/robomimic"
    ["mimicgen_robosuite.patch"]="third_party/mimicgen/robosuite"
    ["mimicgen_robosuite-task-zoo.patch"]="third_party/mimicgen/robosuite-task-zoo"
)

echo "==> Applying patches..."
for patch_file in "${!PATCH_TARGETS[@]}"; do
    target_dir="${PATCH_TARGETS[$patch_file]}"
    patch_path="$PATCHES_DIR/$patch_file"

    if [ ! -f "$patch_path" ]; then
        echo "    [SKIP] $patch_file (not found)"
        continue
    fi

    if [ ! -d "$SCRIPT_DIR/$target_dir" ]; then
        echo "    [SKIP] $patch_file (target $target_dir not found)"
        continue
    fi

    # Check if patch is already applied
    if git -C "$SCRIPT_DIR/$target_dir" apply --check --reverse "$patch_path" 2>/dev/null; then
        echo "    [OK]   $patch_file (already applied)"
    else
        git -C "$SCRIPT_DIR/$target_dir" apply "$patch_path"
        echo "    [OK]   $patch_file -> $target_dir"
    fi
done

echo "==> Done. Activate the conda environment before running:"
echo "    conda activate vlaws"
