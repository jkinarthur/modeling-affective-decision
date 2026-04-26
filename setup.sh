#!/bin/bash
# ============================================================
# setup.sh  –  One-time NVIDIA driver setup for AD-DAN
#
# Target: Deep Learning AMI Neuron (Amazon Linux 2023)
#         g4dn.xlarge  /  NVIDIA Tesla T4 (16 GB)
#
# Run once after launching the instance:
#   bash setup.sh
#
# What this does:
#   1. Installs kernel6.12-devel (avoids kernel-headers conflict)
#   2. Downloads & installs NVIDIA Tesla driver 525.147.05
#      (certified for CUDA 12.x, supports T4)
#   3. Verifies GPU with nvidia-smi
#   4. Restarts Docker so nvidia-container-toolkit picks up the driver
#   5. Disables Docker BuildKit (standalone docker-compose compatibility)
# ============================================================

set -e

DRIVER_VER="525.147.05"
KVER=$(uname -r)

echo "======================================================"
echo "  AD-DAN AWS Setup"
echo "  Kernel : ${KVER}"
echo "  Driver : NVIDIA ${DRIVER_VER}"
echo "======================================================"

# ---- 1. Kernel build dependencies ----------------------------
echo ""
echo "[1/5] Installing kernel build dependencies..."
# The Neuron AMI ships kernel6.12-headers which conflicts with the
# generic 'kernel-headers' / 'kernel-devel' packages.
# kernel6.12-devel is the correct versioned package for this AMI.
sudo dnf install -y "kernel6.12-devel-${KVER}" gcc make perl 2>/dev/null \
  || sudo dnf install -y kernel6.12-devel gcc make perl
echo "      Done."

# ---- 2. Download NVIDIA driver --------------------------------
echo ""
echo "[2/5] Downloading NVIDIA Tesla driver ${DRIVER_VER}..."
cd /tmp
wget -q --show-progress \
  "https://us.download.nvidia.com/tesla/${DRIVER_VER}/NVIDIA-Linux-x86_64-${DRIVER_VER}.run" \
  -O nvidia-driver.run
chmod +x nvidia-driver.run
echo "      Done."

# ---- 3. Install NVIDIA driver ---------------------------------
echo ""
echo "[3/5] Installing NVIDIA driver (takes ~2 minutes)..."
sudo /tmp/nvidia-driver.run \
  --silent \
  --no-cc-version-check \
  --no-questions \
  --kernel-source-path "/usr/src/kernels/${KVER}" 2>&1 \
  | grep -E "^(ERROR|WARNING|Installing)" || true
echo "      Done."

# ---- 4. Verify GPU -------------------------------------------
echo ""
echo "[4/5] Verifying GPU access..."
nvidia-smi

# ---- 5. Configure Docker -------------------------------------
echo ""
echo "[5/5] Configuring Docker..."

# Reconfigure nvidia-container-toolkit runtime (idempotent)
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to pick up new driver libraries
sudo systemctl restart docker

# Set DOCKER_BUILDKIT=0 system-wide so standalone docker-compose
# doesn't require buildx >= 0.17
echo 'export DOCKER_BUILDKIT=0' | sudo tee /etc/profile.d/docker-buildkit.sh > /dev/null
export DOCKER_BUILDKIT=0
echo "      Done."

# ---- Summary -------------------------------------------------
echo ""
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo ""
echo "  cd ~/modeling-affective-decision"
echo ""
echo "  # Build image (~5 min, downloads PyTorch+CUDA base)"
echo "  sudo DOCKER_BUILDKIT=0 docker-compose build"
echo ""
echo "  # Full pipeline"
echo "  sudo docker-compose run --rm addan build"
echo "  sudo docker-compose run --rm addan train-sft"
echo "  sudo docker-compose run --rm addan train-rl"
echo "  sudo docker-compose run --rm addan evaluate"
echo ""
echo "  # Stop instance when done to avoid charges (~\$0.53/hr)"
echo "======================================================"
