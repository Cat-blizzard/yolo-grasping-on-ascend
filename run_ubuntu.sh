#!/bin/bash
# Ubuntu环境启动脚本 - 4类物品分拣系统 (CPU模式)

echo "========================================================================"
echo "  视觉引导机械臂抓取系统 - Ubuntu版本 (CPU模式)"
echo "  目标物品: 苹果、橘子、杯子、瓶子"
echo "========================================================================"
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

echo ""
echo "========================================================================"
echo "运行模式选择:"
echo "  [1] 仅视觉检测 (无需语音/LLM,推荐测试)"
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
