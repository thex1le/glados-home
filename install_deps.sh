#!/bin/bash
# Install GLaDOS dependencies for a specific system.
# Usage: ./install_deps.sh <system>
#   system: gpu, pi4, or pi5
#
# This installs system-level apt packages first, then Python packages
# into the active virtualenv.

set -e

SYSTEM="${1:-}"

if [ -z "$SYSTEM" ]; then
    echo "Usage: ./install_deps.sh <gpu|pi4|pi5>"
    echo ""
    echo "  gpu  - GPU server (Ubuntu, NVIDIA 4090): YOLO, pose, WhisperX, TTS"
    echo "  pi4  - Raspberry Pi 4 B+: servos, LEDs, LCDs, sensors, head camera"
    echo "  pi5  - Raspberry Pi 5: voice, fisheye cameras, IMU"
    exit 1
fi

echo "=== Installing dependencies for: $SYSTEM ==="

# --- System packages (apt) ---
echo ""
echo "--- Installing system packages ---"

# Common across all systems: GStreamer + PyGObject build deps
COMMON_APT=(
    libcairo2-dev
    gobject-introspection
    pkg-config
    gir1.2-gstreamer-1.0
    gstreamer1.0-tools
    gstreamer1.0-plugins-base
    gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad
    gstreamer1.0-rtsp
    gir1.2-gst-rtsp-server-1.0
    python3-dev
    python3-venv
)

# GObject Introspection dev headers: try 2.0 first (newer distros), fall back to 1.0
if apt-cache show libgirepository-2.0-dev &>/dev/null; then
    COMMON_APT+=(libgirepository-2.0-dev)
else
    COMMON_APT+=(libgirepository1.0-dev)
fi

case "$SYSTEM" in
    gpu)
        APT_PACKAGES=(
            "${COMMON_APT[@]}"
            mosquitto
            mosquitto-clients
            ffmpeg
            espeak-ng
            libgstreamer1.0-dev
            libgstreamer-plugins-base1.0-dev
        )
        PIP_REQUIREMENTS="requirements_gpu.txt"
        ;;
    pi4)
        APT_PACKAGES=(
            "${COMMON_APT[@]}"
            python3-libcamera
            python3-picamera2
            libatlas-base-dev
            i2c-tools
        )
        PIP_REQUIREMENTS="requirements_pi4.txt"
        ;;
    pi5)
        APT_PACKAGES=(
            "${COMMON_APT[@]}"
            python3-libcamera
            python3-picamera2
            libatlas-base-dev
            i2c-tools
            libasound2-dev
            portaudio19-dev
            ffmpeg
        )
        PIP_REQUIREMENTS="requirements_pi5.txt"
        ;;
    *)
        echo "Unknown system: $SYSTEM"
        echo "Use: gpu, pi4, or pi5"
        exit 1
        ;;
esac

sudo apt update
sudo apt install -y "${APT_PACKAGES[@]}"

# --- Python packages (pip) ---
echo ""
echo "--- Installing Python packages ---"

if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No virtualenv active. Install into system Python? (y/N)"
    read -r REPLY
    if [ "$REPLY" != "y" ] && [ "$REPLY" != "Y" ]; then
        echo "Aborted. Activate your virtualenv first:"
        echo "  source <venv>/bin/activate"
        exit 1
    fi
fi

if [ "$SYSTEM" = "gpu" ]; then
    echo ""
    echo "NOTE: Install PyTorch with CUDA separately first if not already installed:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    echo ""
fi

pip install -r "$PIP_REQUIREMENTS"

# --- Build OpenCV from source with GStreamer + CUDA (GPU server) ---
if [ "$SYSTEM" = "gpu" ]; then
    echo ""
    echo "--- Building OpenCV from source with GStreamer + CUDA ---"
    echo "  This enables nvh264dec GPU decode for RTSP camera streams."
    echo "  Takes 30-60 minutes. Grab a coffee."
    echo ""

    OPENCV_VERSION="4.13.0"
    BUILD_DIR="/tmp/opencv_build"
    VENV_PYTHON=$(which python3)
    SITE_PACKAGES=$($VENV_PYTHON -c "import site; print(site.getsitepackages()[0])")

    # Build deps for OpenCV + cuDNN for CUDA DNN backend
    sudo apt-get install -y \
        build-essential cmake git \
        libavcodec-dev libavformat-dev libswscale-dev \
        libv4l-dev libxvidcore-dev libx264-dev \
        libjpeg-dev libpng-dev libtiff-dev \
        libatlas-base-dev gfortran python3-numpy \
        libcudnn9-dev-cuda-12 libcudnn9-cuda-12

    # Remove pip opencv (conflicts with source build)
    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

    # Clone OpenCV source
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    git clone --depth 1 --branch "$OPENCV_VERSION" https://github.com/opencv/opencv.git
    git clone --depth 1 --branch "$OPENCV_VERSION" https://github.com/opencv/opencv_contrib.git

    # Configure with GStreamer + CUDA
    mkdir -p opencv/build
    cd opencv/build
    cmake .. \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH="$BUILD_DIR/opencv_contrib/modules" \
        -D PYTHON3_EXECUTABLE="$VENV_PYTHON" \
        -D PYTHON3_INCLUDE_DIR=$($VENV_PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))") \
        -D PYTHON3_PACKAGES_PATH="$SITE_PACKAGES" \
        -D BUILD_opencv_python3=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_CUDA=ON \
        -D CUDA_ARCH_BIN=8.9 \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D WITH_CUBLAS=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_V4L=ON \
        -D WITH_OPENGL=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_java=OFF \
        -D BUILD_opencv_apps=OFF

    # Build and install
    NPROC=$(nproc)
    echo "Building with $NPROC cores..."
    make -j"$NPROC"
    sudo make install
    sudo ldconfig

    # Verify
    echo ""
    echo "--- OpenCV build verification ---"
    $VENV_PYTHON -c "
import cv2
print(f'OpenCV version: {cv2.__version__}')
info = cv2.getBuildInformation()
for line in info.split('\n'):
    ll = line.strip().lower()
    if any(k in ll for k in ['gstreamer', 'cuda', 'ffmpeg']):
        print(f'  {line.strip()}')
"
    cd "$OLDPWD"
fi

# --- Dev/test dependencies (all systems) ---
echo ""
echo "--- Installing test dependencies ---"
pip install pytest flexmock

# --- MQTT broker setup (GPU server only) ---
if [ "$SYSTEM" = "gpu" ]; then
    echo ""
    echo "--- Configuring MQTT broker (mosquitto) ---"
    # Allow connections from Pi4/Pi5 on the network (mosquitto 2.x defaults to localhost only)
    if [ ! -f /etc/mosquitto/conf.d/glados.conf ]; then
        echo -e "listener 1883\nallow_anonymous true" | sudo tee /etc/mosquitto/conf.d/glados.conf > /dev/null
        echo "Created /etc/mosquitto/conf.d/glados.conf (listen on all interfaces, port 1883)"
    else
        echo "MQTT config /etc/mosquitto/conf.d/glados.conf already exists, skipping"
    fi
    sudo systemctl enable mosquitto
    sudo systemctl restart mosquitto
    echo "Mosquitto broker enabled and started"
fi

echo ""
echo "=== Done. Run 'python -m pytest Tests/ -v' to verify. ==="
