#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统诊断工具 - 检查步骤六卡住的原因
"""

import os
import sys
import subprocess
import platform

print("="*70)
print("🔍 机械臂系统诊断工具")
print("="*70)

def check_ros2_environment():
    """检查ROS2环境"""
    print("\n📦 1. 检查ROS2环境")
    print("-"*70)
    
    try:
        # 检查ROS2是否已安装
        result = subprocess.run(['ros2', '--version'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print(f"✅ ROS2已安装: {result.stdout.strip()}")
        else:
            print("❌ ROS2未安装或配置错误")
            return False
    except Exception as e:
        print(f"❌ ROS2检查失败: {e}")
        return False
    
    # 检查ROS2服务列表
    print("\n📋 检查ROS2服务列表...")
    try:
        result = subprocess.run(['ros2', 'service', 'list'], capture_output=True, text=True, timeout=5)
        services = result.stdout.strip().split('\n')
        print(f"   发现 {len(services)} 个服务:")
        
        # 查找trial_service
        if 'trial_service' in result.stdout or '/trial_service' in result.stdout:
            print("✅ 逆运动学服务 'trial_service' 已就绪")
        else:
            print("❌ 未找到 'trial_service' 服务!")
            print("💡 请启动服务: ros2 run dofbot_info kinemarics_server")
            print("\n所有服务列表:")
            for svc in services[:10]:  # 只显示前10个
                print(f"   - {svc}")
            return False
    except Exception as e:
        print(f"❌ 服务列表检查失败: {e}")
        return False
    
    return True

def check_serial_port():
    """检查机械臂串口连接"""
    print("\n🔌 2. 检查机械臂串口连接")
    print("-"*70)
    
    if platform.system() == "Windows":
        # Windows: 查找COM端口
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            if ports:
                print("✅ 检测到串口设备:")
                for port in ports:
                    print(f"   - {port.device}: {port.description}")
                return True
            else:
                print("❌ 未检测到串口设备 (COM端口)")
                return False
        except ImportError:
            print("⚠️ 未安装pyserial,跳过串口检查")
            print("   安装: pip install pyserial")
    else:
        # Linux: 查找/dev/ttyUSB*
        result = subprocess.run(['ls', '/dev/ttyUSB*'], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            ports = result.stdout.strip().split('\n')
            print(f"✅ 检测到USB串口: {ports}")
            return True
        else:
            print("❌ 未检测到USB串口 (/dev/ttyUSB*)")
            print("💡 请检查:")
            print("   1. 机械臂是否已连接")
            print("   2. 驱动是否已安装")
            print("   3. 用户权限: sudo usermod -aG dialout $USER")
            return False

def check_python_packages():
    """检查Python依赖包"""
    print("\n📚 3. 检查Python依赖包")
    print("-"*70)
    
    packages = {
        'rclpy': 'ROS2 Python客户端',
        'Arm_Lib': '机械臂控制库',
        'cv2': 'OpenCV',
        'numpy': 'NumPy'
    }
    
    all_ok = True
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            print(f"✅ {pkg:15s} - {desc}")
        except ImportError:
            print(f"❌ {pkg:15s} - {desc} (未安装)")
            all_ok = False
    
    return all_ok

def check_offset_file():
    """检查offset.txt文件"""
    print("\n📐 4. 检查坐标偏移文件")
    print("-"*70)
    
    # 根据系统查找offset.txt
    if platform.system() == "Windows":
        offset_path = r"d:\robocode\ros2_robot_arm\ros2_ws\src\dofbot_garbage_yolov5\dofbot_garbage_yolov5\config\offset.txt"
    else:
        offset_path = "/home/HwHiAiUser/robocode_ld3/ros2_robot_arm/ros2_ws/src/dofbot_garbage_yolov5/dofbot_garbage_yolov5/config/offset.txt"
    
    if os.path.exists(offset_path):
        print(f"✅ offset.txt 存在: {offset_path}")
        with open(offset_path, 'r') as f:
            content = f.readlines()
            print(f"   y_offset = {content[0].strip()}")
            print(f"   x_offset = {content[1].strip()}")
        return True
    else:
        print(f"❌ offset.txt 不存在: {offset_path}")
        return False

def test_ros2_service():
    """测试ROS2服务调用"""
    print("\n🧪 5. 测试ROS2逆运动学服务")
    print("-"*70)
    
    try:
        # 尝试调用服务
        print("📝 发送测试请求: (x=0.05, y=0.20, z=0.0)")
        cmd = [
            'ros2', 'service', 'call', '/trial_service',
            'dofbot_info/srv/Kinemarics',
            '{tar_x: 0.05, tar_y: 0.20, tar_z: 0.0, kin_name: "ik"}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ 服务调用成功!")
            print(f"   响应: {result.stdout[:200]}")  # 只显示前200字符
            return True
        else:
            print(f"❌ 服务调用失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ 服务调用超时(10秒)")
        print("💡 可能原因:")
        print("   - 服务未启动")
        print("   - 求解器卡死")
        print("   - 目标坐标超出工作空间")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    results = {
        'ROS2环境': check_ros2_environment(),
        '串口连接': check_serial_port(),
        'Python依赖': check_python_packages(),
        'offset文件': check_offset_file()
    }
    
    # 如果ROS2可用,测试服务
    if results['ROS2环境']:
        results['ROS2服务'] = test_ros2_service()
    
    # 总结
    print("\n" + "="*70)
    print("📊 诊断总结")
    print("="*70)
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {name:15s}: {'正常' if status else '异常'}")
    
    # 给出建议
    print("\n💡 问题排查建议:")
    print("-"*70)
    
    if not results.get('ROS2环境', False):
        print("1️⃣ ROS2环境未就绪:")
        print("   - 执行: source /opt/ros/humble/setup.bash")
        print("   - 检查: echo $ROS_DISTRO")
        
    if not results.get('ROS2服务', True):
        print("2️⃣ 逆运动学服务未响应:")
        print("   - 启动服务: cd ros2_ws && source install/setup.bash")
        print("   - 运行节点: ros2 run dofbot_info kinemarics_server")
        
    if not results.get('串口连接', False):
        print("3️⃣ 机械臂串口未连接:")
        print("   - 检查USB线缆")
        print("   - 检查设备权限 (Linux)")
        print("   - 尝试重新插拔")
    
    print("\n📋 查看完整日志的方法:")
    print("   运行程序时会在终端实时输出详细日志")
    print("   步骤六卡住时,按 Ctrl+C 中断,查看最后几行日志")
    print("="*70)

if __name__ == "__main__":
    main()
