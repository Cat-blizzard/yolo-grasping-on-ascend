#!/bin/bash
# Ubuntu环境启动脚本 - 4类物品分拣系统 (CPU模式)

echo "========================================================================"
echo "  视觉引导机械臂抓取系统 - Ubuntu版本 (CPU模式)"
echo "  目标物品: 苹果、橘子、杯子、瓶子"
echo "========================================================================"
echo ""

# 检查ROS2环境
echo "检查ROS2环境..."
if [ -f "/opt/ros/humble/setup.bash" ]; then
    echo "✓ ROS2 Humble 已安装"
    source /opt/ros/humble/setup.bash
    echo "✓ ROS2环境已加载"
else
    echo "⚠️  ROS2未安装,机械臂功能将被禁用"
fi

# 加载工作空间
WS_PATH="/home/HwHiAiUser/robocode_ld3/ros2_robot_arm/ros2_ws"
if [ -f "$WS_PATH/install/setup.bash" ]; then
    cd "$WS_PATH"
    source install/setup.bash
    echo "✓ ROS2工作空间已加载"
    cd /home/HwHiAiUser/robocode_ld3
else
    echo "⚠️  ROS2工作空间未编译,请先运行: cd $WS_PATH && colcon build"
fi

echo ""

# 检测Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python版本: $python_version"

# 检测依赖
echo ""
echo "检测依赖包..."
dependencies=("torch" "ultralytics" "opencv-python" "numpy")
missing_deps=()

for dep in "${dependencies[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo "✓ $dep 已安装"
    else
        echo "✗ $dep 未安装"
        missing_deps+=("$dep")
    fi
done

# 如果有缺失依赖,提示安装
if [ ${#missing_deps[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  检测到缺失依赖,请运行以下命令安装:"
    echo "   pip3 install ${missing_deps[*]}"
    echo ""
    read -p "是否现在安装? (y/n): " install_choice
    if [ "$install_choice" == "y" ]; then
        pip3 install ${missing_deps[*]}
    else
        echo "已取消,请手动安装依赖后重新运行"
        exit 1
    fi
fi

# 检查ROS2服务
echo ""
echo "检查ROS2逆运动学服务..."
if command -v ros2 &> /dev/null; then
    if ros2 service list 2>/dev/null | grep -q "trial_service"; then
        echo "✓ 逆运动学服务已运行"
    else
        echo "⚠️  逆运动学服务未运行"
        echo ""
        read -p "是否需要启动逆运动学服务? (y/n): " start_service
        if [ "$start_service" == "y" ]; then
            echo "在新终端运行以下命令启动服务:"
            echo "  source /opt/ros/humble/setup.bash"
            echo "  cd $WS_PATH"
            echo "  source install/setup.bash"
            echo "  ros2 run dofbot_info kinemarics_server"
            echo ""
            read -p "服务启动后按Enter继续..."
        fi
    fi
fi

echo ""
echo "========================================================================"
echo "运行模式选择:"
echo "  [1] 仅视觉检测 (无需语音/LLM/机械臂,推荐测试)"
echo "  [2] 完整系统 - 单次执行 (语音+LLM+视觉+机械臂)"
echo "  [3] 完整系统 - 持续运行"
echo "  [0] 退出"
echo "========================================================================"
read -p "请选择运行模式 [0-3]: " mode_choice

case $mode_choice in
    1)
        echo ""
        echo "▶️  启动仅视觉检测模式..."
        python3 voice_guided_robot_system.py --mode vision_only
        ;;
    2)
        echo ""
        echo "▶️  启动完整系统 (单次执行)..."
        python3 voice_guided_robot_system.py --mode once
        ;;
    3)
        echo ""
        echo "▶️  启动完整系统 (持续运行)..."
        python3 voice_guided_robot_system.py --mode continuous
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "程序已结束"
echo "========================================================================"
