#!/bin/bash
# ============================================================
# setup.sh  –  One-time NVIDIA driver + Docker setup for AD-DAN
#
# Target: Amazon Linux 2023 kernel-6.1  /  g4dn.xlarge  /  NVIDIA T4
#
# Run once after launching the instance:
#   bash setup.sh
# ============================================================

set -e

KVER=$(uname -r)

echo "======================================================"
echo "  AD-DAN AWS Setup  (Amazon Linux 2023 kernel-6.1)"
echo "  Kernel : ${KVER}"
echo "======================================================"

# ---- 1. Add NVIDIA CUDA repo ---------------------------------
echo ""
echo "[1/5] Adding NVIDIA CUDA repository..."
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf clean all
echo "      Done."

# ---- 2. Install NVIDIA driver --------------------------------
echo ""
echo "[2/5] Installing NVIDIA driver..."
sudo dnf install -y kernel-devel-$(uname -r) gcc make 2>/dev/null \
    || sudo dnf install -y kernel-devel gcc make
sudo dnf module install -y nvidia-driver:latest-dkms 2>/dev/null \
    || sudo dnf install -y nvidia-driver-latest-dkms
echo "      Done."

# ---- 3. Verify GPU -------------------------------------------
echo ""
echo "[3/5] Verifying GPU access..."
nvidia-smi

# ---- 4. Install Docker CE ------------------------------------
echo ""
echo "[4/5] Installing Docker + nvidia-container-toolkit..."
sudo dnf install -y docker
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
echo "      Done."

# ---- 5. Install docker-compose v2 plugin ---------------------
echo ""
echo "[5/5] Installing docker-compose v2..."
COMPOSE_VER="v2.27.0"
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -fsSL "https://github.com/docker/compose/releases/download/${COMPOSE_VER}/docker-compose-linux-x86_64" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

echo ""
echo "======================================================"
echo "  Setup complete! Next:"
echo "  cd ~/modeling-affective-decision"
echo "  DOCKER_BUILDKIT=0 docker compose build"
echo "  docker compose run --rm addan train-sft"
echo "======================================================"

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
