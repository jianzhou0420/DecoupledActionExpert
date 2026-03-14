#!/bin/bash
# =============================================================================
# DecoupledActionExpert Environment Installation
# =============================================================================
# Usage: ./env_install.sh
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="DecoupledActionExpert"

echo "=============================================="
echo "  DecoupledActionExpert Environment Setup"
echo "=============================================="
echo ""

# Check if environment already exists
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Environment '$ENV_NAME' already exists."
    echo "  [r] Remove and recreate"
    echo "  [u] Update existing"
    echo "  [s] Skip"
    read -p "Choice [r/u/S]: " choice
    case "$choice" in
        r|R) ACTION="create" ;;
        u|U) ACTION="update" ;;
        *)   echo "Skipped."; exit 0 ;;
    esac
else
    ACTION="create"
fi

# Create or update conda environment
case "$ACTION" in
    create)
        if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
            echo "Removing existing environment..."
            conda env remove -n "$ENV_NAME" -y
        fi
        echo "Creating conda environment from environment.yaml..."
        mamba env create -f "$PROJECT_ROOT/environment.yaml" -n "$ENV_NAME" -y
        ;;
    update)
        echo "Updating existing conda environment..."
        mamba env update -f "$PROJECT_ROOT/environment.yaml" -n "$ENV_NAME" -y
        ;;
esac

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Apply patches to third-party submodules
echo "=== Applying patches to third-party dependencies ==="
bash "$PROJECT_ROOT/setup.sh"

# Install third-party packages (editable, --no-deps to use conda-managed versions)
echo "=== Installing third-party packages (editable) ==="
pip install --no-deps -e "$PROJECT_ROOT/third_party/lerobot"
pip install --no-deps -e "$PROJECT_ROOT/third_party/libero"
pip install --no-deps -e "$PROJECT_ROOT/third_party/mimicgen/robosuite"
pip install --no-deps -e "$PROJECT_ROOT/third_party/mimicgen/robosuite-task-zoo"
pip install --no-deps -e "$PROJECT_ROOT/third_party/mimicgen/robomimic"
pip install --no-deps -e "$PROJECT_ROOT/third_party/mimicgen/mimicgen"

# Install this package (editable)
echo "=== Installing DecoupledActionExpert (editable) ==="
pip install -e "$PROJECT_ROOT"

# Verify
echo "=== Verifying installation ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import vlaworkspace; print('vlaworkspace OK')"

echo ""
echo "=============================================="
echo "  Installation complete!"
echo "  Activate with: conda activate $ENV_NAME"
echo "=============================================="
