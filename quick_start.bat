@echo off
chcp 65001 >nul
echo ================================================================================
echo           视觉引导机械臂抓取系统 - 快速启动脚本
echo ================================================================================
echo.

:menu
echo 请选择运行模式:
echo.
echo [1] 完整系统 - 单次执行 (需要ROS2 + 机械臂)
echo [2] 完整系统 - 持续运行 (需要ROS2 + 机械臂)
echo [3] 测试模式 - 无机械臂 (仅测试语音+视觉+匹配)
echo [4] 仅测试语音识别
echo [5] 仅测试视觉检测
echo [6] 查看系统文档
echo [0] 退出
echo.

set /p choice=请输入选项 [0-6]: 

if "%choice%"=="1" goto full_once
if "%choice%"=="2" goto full_continuous
if "%choice%"=="3" goto test_integration
if "%choice%"=="4" goto test_voice
if "%choice%"=="5" goto test_vision
if "%choice%"=="6" goto show_docs
if "%choice%"=="0" goto end
goto menu

:full_once
echo.
echo ▶️ 启动完整系统 (单次执行)...
echo.
python voice_guided_robot_system.py --mode once
pause
goto menu

:full_continuous
echo.
echo ▶️ 启动完整系统 (持续运行)...
echo 提示: 按 Ctrl+C 可中断退出
echo.
python voice_guided_robot_system.py --mode continuous
pause
goto menu

:test_integration
echo.
echo ▶️ 启动测试模式 (无机械臂)...
echo.
python test_integration.py
pause
goto menu

:test_voice
echo.
echo ▶️ 测试语音识别...
echo 请在5秒内说话...
echo.
python -c "import sys; sys.path.insert(0, 'mindyolo-master/demo'); from recognize_voice import asr_recognize; print('识别结果:', asr_recognize(5.0))"
pause
goto menu

:test_vision
echo.
echo ▶️ 测试视觉检测...
echo 正在启动摄像头并执行检测...
echo.
python -c "import cv2; cap=cv2.VideoCapture(0); ret,img=cap.read(); cap.release(); print('摄像头状态:', '✅ 正常' if ret else '❌ 失败'); cv2.imshow('Camera Test', img) if ret else None; cv2.waitKey(2000); cv2.destroyAllWindows()"
pause
goto menu

:show_docs
echo.
echo ================================================================================
echo 系统文档列表:
echo ================================================================================
echo.
echo [1] SYSTEM_USAGE.txt      - 使用说明文档
echo [2] ARCHITECTURE.txt      - 架构设计文档
echo [3] voice_guided_robot_system.py - 主程序源码
echo [4] test_integration.py   - 测试脚本源码
echo.
echo 请使用文本编辑器打开查看
echo.
pause
goto menu

:end
echo.
echo 👋 感谢使用!系统已退出
echo.
exit
