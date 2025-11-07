# 🤖 Voice-Guided Robot Sorting System

<div align="center">

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![YOLO](https://img.shields.io/badge/YOLO-v3--v11-yellow.svg)](https://github.com/ultralytics/ultralytics)

**An intelligent robotic sorting system integrating voice recognition, computer vision, and robotic manipulation**

[English](README.md) | [中文](README_CN.md)

</div>

---

### 📖 Overview

This project implements an end-to-end robotic sorting system that combines:
- **Voice Recognition**: Natural language command processing
- **LLM Integration**: Intent extraction using large language models
- **Computer Vision**: YOLO-based object detection (v3-v11 supported)
- **Robot Control**: ROS2-powered robotic arm manipulation

**Supported Objects**: Apple 🍎 | Orange 🍊 | Cup ☕ | Bottle 🍾

### ✨ Key Features

- 🎯 **Voice-to-Action Pipeline**: Speak → Detect → Grasp → Sort
- 🚀 **Hardware Acceleration**: NPU (Ascend 310B) / GPU (CUDA) support
- 🔄 **Multi-Mode Operation**: Vision-only testing / Single execution / Continuous loop
- 📦 **Modular Architecture**: Easy to extend and customize
- 🛠️ **Production Ready**: Comprehensive logging and error handling

### 🎬 Demo Workflow

```
User: "Please grab an apple"
  ↓
[Voice Recognition] → Transcribe to text
  ↓
[LLM Parser] → Extract intent: target="apple"
  ↓
[YOLO Detection] → Locate apple in camera frame
  ↓
[Coordinate Mapping] → Pixel → Robot base coordinates
  ↓
[Inverse Kinematics] → Calculate joint angles
  ↓
[Robot Execution] → Grasp → Move to sorting bin → Release
```

### 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Voice Command                        │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│  Voice Recognition Module (recognize_voice.py)               │
│  ├─ Audio Input (Microphone)                                 │
│  └─ Speech-to-Text Conversion                                │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│  LLM Intent Parser (LLM意图识别.py)                           │
│  ├─ Natural Language Understanding                            │
│  └─ Target Object Extraction                                  │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│  Vision Perception Module (VisionPerception class)           │
│  ├─ YOLO Detection (v3-v11 supported)                        │
│  ├─ Backend: NPU/GPU/CPU auto-selection                      │
│  └─ Object Localization                                       │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│  Decision Layer (DecisionLayer class)                        │
│  └─ Match voice target with visual detections                │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│  Coordinate Mapping (CoordinateMapper class)                 │
│  └─ Pixel coordinates → Robot base frame                     │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│  Robot Control (RobotArmController class)                    │
│  ├─ ROS2 Service: Inverse Kinematics (trial_service)         │
│  ├─ Motion Planning: MoveIt2                                  │
│  └─ Execution: Grasp → Transport → Release                   │
└──────────────────────────────────────────────────────────────┘
```

**Directory Structure**:
```
├── mindyolo-master/              # YOLO model training & inference
│   ├── configs/                  # Model configs (YOLOv3-v11)
│   ├── demo/                     # Demo scripts
│   │   ├── LLM意图识别.py        # LLM integration
│   │   ├── recognize_voice.py    # Voice recognition
│   │   └── npupredict.py         # NPU-accelerated inference
│   └── deploy/                   # Model deployment tools
│
├── ros2_robot_arm/               # Robot control system
│   └── ros2_ws/                  # ROS2 workspace
│       └── src/
│           ├── dofbot_moveit/    # Motion planning (IK server)
│           ├── dofbot_info/      # Robot messages & services
│           └── dofbot_garbage_yolov5/  # Legacy detection module
│
├── voice_guided_robot_system.py  # Main program
├── debug_check.py                # Diagnostic tool
└── run_ubuntu.sh                 # Launch script
```

### 🔧 Requirements

#### Hardware
- **OS**: Ubuntu 20.04 / 22.04 (recommended)
- **Robot**: Dofbot 5-DOF arm or compatible
- **Camera**: USB/CSI camera (640×480 or higher)
- **Microphone**: For voice input
- **Optional**: Huawei Ascend NPU (310B) for acceleration

#### Software
| Component | Version | Required |
|-----------|---------|----------|
| Python | 3.8+ | ✅ |
| ROS2 | Humble | ✅ |
| PyTorch | 1.10+ | ✅ (CPU/GPU) |
| Ultralytics | Latest | ✅ |
| OpenCV | 4.x | ✅ |
| MindSpore | 2.x | ⚪ (for NPU) |
| CUDA | 11.x+ | ⚪ (for GPU) |

### 🚀 Quick Start

#### 1️⃣ Install Dependencies

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/voice-guided-robot-sorting.git
cd voice-guided-robot-sorting

# Install Python packages
pip3 install -r requirements.txt

# Install ROS2 Humble (if not installed)
# Ubuntu 22.04:
sudo apt update
sudo apt install ros-humble-desktop

# Install robot control library
cd ros2_robot_arm/0.py_install
pip3 install .
```

#### 2️⃣ Configure API Keys

**Important**: Set up your API credentials before running the system.

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and fill in your API keys:
# - XFYUN_APPID: Get from https://console.xfyun.cn/
# - XFYUN_API_KEY: iFlytek ASR API key
# - XFYUN_API_SECRET: iFlytek ASR secret
# - DOUBAO_API_KEY: Get from https://console.volcengine.com/ark
```

**Load environment variables**:

```bash
# Linux/Mac
source .env
# Or
export $(cat .env | xargs)

# Windows PowerShell
Get-Content .env | ForEach-Object {
    $var = $_.Split('=')
    [Environment]::SetEnvironmentVariable($var[0], $var[1])
}
```

> **🔒 Security**: Never commit `.env` file to Git! It's already in `.gitignore`.

#### 3️⃣ Build ROS2 Workspace

```bash
cd ros2_robot_arm/ros2_ws

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Build workspace
colcon build --symlink-install

# Source workspace
source install/setup.bash
```

> **⚠️ Important**: Always run `source /opt/ros/humble/setup.bash` before building!

#### 4️⃣ Launch System

**Option A: Vision-Only Mode** (No robot required)
```bash
./run_ubuntu.sh
# Select [1] Vision Detection Only
```

**Option B: Full System** (Requires robot hardware)

*Terminal 1 - Start ROS2 service:*
```bash
source /opt/ros/humble/setup.bash
cd ros2_robot_arm/ros2_ws
source install/setup.bash
ros2 run dofbot_moveit dofbot_server
```

*Terminal 2 - Run main program:*
```bash
source /opt/ros/humble/setup.bash
cd ros2_robot_arm/ros2_ws
source install/setup.bash
cd ../../
python3 voice_guided_robot_system.py
```

**Option C: Use Launch Script**
```bash
chmod +x run_ubuntu.sh
./run_ubuntu.sh
# Select mode: [1] Vision | [2] Single Run | [3] Continuous
```

### 💬 Usage Examples

#### Voice Commands (Natural Language)

```
User: "Please grab an apple"          → System: Detects & picks apple
User: "帮我拿一个橘子"                  → System: Detects & picks orange  
User: "I need a cup"                   → System: Detects & picks cup
User: "把瓶子放到篮子里"               → System: Detects & picks bottle
```

**Supported languages**: English, Chinese (中文)

#### Supported Objects

| Object | English | Chinese | COCO Class |
|--------|---------|---------|------------|
| 🍎 | Apple | 苹果 | `apple` |
| 🍊 | Orange | 橘子/橙子 | `orange` |
| ☕ | Cup | 杯子/水杯 | `cup` |
| 🍾 | Bottle | 瓶子 | `bottle` |

#### Sorting Bins Configuration

Each object is sorted to a predefined location:
```python
SORT_POSITIONS = {
    "apple":  Position 1 (Front-Left)   # [45°, 50°, 20°, 60°, 265°]
    "orange": Position 2 (Back-Left)    # [27°, 75°, 0°, 50°, 265°]
    "cup":    Position 3 (Back-Right)   # [147°, 75°, 0°, 50°, 265°]
    "bottle": Position 4 (Front-Right)  # [133°, 50°, 20°, 60°, 265°]
}
```

### ⚙️ Configuration

#### System Config (`system_config.py`)

```python
# Camera settings
CAMERA_ID = 0              # USB camera index
RESOLUTION = (640, 480)    # Image size

# Detection parameters
CONF_THRESHOLD = 0.5       # Confidence threshold
IOU_THRESHOLD = 0.65       # NMS IoU threshold

# Robot settings
ROBOT_SPEED = 1000         # Servo speed (ms)
GRASP_ANGLE = 130          # Gripper close angle

# Acceleration
DEVICE = "auto"            # auto | npu | cuda | cpu
```

#### YOLO Model Selection

Supported models in `mindyolo-master/configs/`:
- **YOLOv3** - Classic baseline
- **YOLOv5** - Fast & accurate (s/m/l/x variants)
- **YOLOv7** - High performance
- **YOLOv8** - Latest Ultralytics (recommended)
- **YOLOv9/v10/v11** - Cutting edge
- **YOLOX** - Anchor-free alternative

To switch models, edit `voice_guided_robot_system.py`:
```python
config = {
    "model_path_pt": "path/to/yolov8s.pt",  # Change model here
    # ...
}
```

### 🔍 Troubleshooting

<details>
<summary><b>❌ ROS2 Service Timeout</b></summary>

**Error**: `❌ [失败] 逆运动学服务超时(5秒内未响应)`

**Solution**: Start the inverse kinematics service:
```bash
source /opt/ros/humble/setup.bash
cd ros2_robot_arm/ros2_ws && source install/setup.bash
ros2 run dofbot_moveit dofbot_server
```

Verify service is running:
```bash
ros2 service list | grep trial_service
```
</details>

<details>
<summary><b>❌ Module Import Error</b></summary>

**Error**: `ModuleNotFoundError: No module named 'mindyolo'`

**Solution**: Install mindyolo package:
```bash
cd mindyolo-master
pip3 install -e .
```
</details>

<details>
<summary><b>❌ ROS2 Build Failed</b></summary>

**Error**: `Could not find a package configuration file provided by "ament_cmake"`

**Solution**: Source ROS2 environment before building:
```bash
source /opt/ros/humble/setup.bash
cd ros2_robot_arm/ros2_ws
colcon build --symlink-install
```
</details>

<details>
<summary><b>❌ Camera Not Found</b></summary>

**Error**: `❌ 无法打开摄像头`

**Solution**: 
1. Check camera connection: `ls /dev/video*`
2. Test camera: `cheese` or `v4l2-ctl --list-devices`
3. Change camera ID in config: `CAMERA_ID = 1` (try 0, 1, 2...)
</details>

<details>
<summary><b>❌ Voice Recognition Not Working</b></summary>

**Solution**:
- Check microphone permissions
- Test microphone: `arecord -d 5 test.wav && aplay test.wav`
- Install audio libraries: `sudo apt install portaudio19-dev`
</details>

<details>
<summary><b>🛠️ Diagnostic Tool</b></summary>

Run automated system check:
```bash
python3 debug_check.py
```

This checks:
- ✅ ROS2 environment
- ✅ Service availability  
- ✅ Serial port connection
- ✅ Python dependencies
</details>

### 🧪 Development

#### Testing

```bash
# Test vision detection only
python3 test_4class_detection.py

# Test full system integration
python3 test_integration.py

# Run diagnostic check
python3 debug_check.py
```

#### Model Training

To train custom YOLO models:
```bash
cd mindyolo-master

# Prepare dataset (COCO format)
# Edit config: configs/yolov8/yolov8s.yaml

# Train model
python train.py --config configs/yolov8/yolov8s.yaml \
                --data configs/coco.yaml \
                --epochs 100

# Export for deployment
python export.py --config configs/yolov8/yolov8s.yaml \
                 --weight runs/train/weights/best.ckpt
```

See `mindyolo-master/GETTING_STARTED.md` for details.

#### Adding New Object Classes

1. **Update mapping** in `voice_guided_robot_system.py`:
```python
OBJECT_MAPPING = {
    "苹果": "apple",
    "your_object": "coco_class_name",  # Add here
}
```

2. **Add sorting position**:
```python
SORTING_POSITIONS = {
    "coco_class_name": [j1, j2, j3, j4, j5],  # Joint angles
}
```

3. **Update LLM prompts** in `demo/LLM意图识别.py`

### 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Vision** | YOLO v3-v11, YOLOX (MindSpore/PyTorch/Ultralytics) |
| **Voice** | Speech Recognition Library |
| **NLP** | Large Language Model (LLM) Integration |
| **Robot Control** | ROS2 Humble + MoveIt2 |
| **Kinematics** | Orocos KDL |
| **Acceleration** | Huawei Ascend NPU / NVIDIA CUDA |
| **Framework** | Python 3.8+, C++ 14 |

### 📚 Documentation

- [`ARCHITECTURE.txt`](ARCHITECTURE.txt) - System architecture details
- [`SYSTEM_USAGE.txt`](SYSTEM_USAGE.txt) - Usage guide
- [`UBUNTU_SETUP.txt`](UBUNTU_SETUP.txt) - Ubuntu setup instructions
- [`完整部署指南.txt`](完整部署指南.txt) - Complete deployment guide (中文)
- [`4类物品分拣系统更新说明.md`](4类物品分拣系统更新说明.md) - Update log (中文)

### 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

### 📄 License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection models
- [MindSpore](https://www.mindspore.cn/) - Deep learning framework
- [ROS2](https://docs.ros.org/) - Robot Operating System
- [Orocos KDL](https://www.orocos.org/kdl.html) - Kinematics library

### 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/voice-guided-robot-sorting/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/voice-guided-robot-sorting/discussions)

---

<div align="center">

**⭐ If you find this project helpful, please give it a star! ⭐**

**💡 First-time users: Start with Vision-Only mode to verify detection before enabling full system**

Made with ❤️ by the Robot Vision Team

</div>
