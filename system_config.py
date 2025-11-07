#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统配置文件 - 集中管理所有可配置参数
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================
ROBOCODE_ROOT = Path(r"d:\robocode")
MINDYOLO_ROOT = ROBOCODE_ROOT / "mindyolo-master"
ROS2_ROOT = ROBOCODE_ROOT / "ros2_robot_arm" / "ros2_ws"

# 模型文件
YOLOV8_MODEL_PATH = MINDYOLO_ROOT / "yolov8s_coco.mindir"

# 配置文件
OFFSET_CONFIG_PATH = ROS2_ROOT / "src" / "dofbot_garbage_yolov5" / "dofbot_garbage_yolov5" / "config" / "offset.txt"
DP_BIN_PATH = ROS2_ROOT / "src" / "dofbot_garbage_yolov5" / "dofbot_garbage_yolov5" / "config" / "dp.bin"

# ==================== 硬件配置 ====================
# 摄像头
CAMERA_CONFIG = {
    "camera_id": 0,              # 摄像头ID (0=默认USB摄像头)
    "width": 1280,               # 采集分辨率宽度
    "height": 720,               # 采集分辨率高度
    "fourcc": "MJPG",            # 编码格式
    "process_width": 640,        # 处理分辨率宽度
    "process_height": 480        # 处理分辨率高度
}

# 机械臂
ROBOT_ARM_CONFIG = {
    "init_position": [90, 135],  # 初始位置 [joint1, joint2]
    "gripper_close_angle": 130,  # 夹爪闭合角度
    "gripper_open_angle": 30,    # 夹爪打开角度
    "uart_baudrate": 115200,     # 串口波特率
    "movement_speed": 1000       # 运动速度(ms)
}

# ==================== 算法配置 ====================
# 语音识别 - API密钥从环境变量读取
# 设置方法:
#   Linux/Mac: export XFYUN_APPID="your-appid"
#   Windows: set XFYUN_APPID=your-appid
ASR_CONFIG = {
    "appid": os.getenv("XFYUN_APPID", "YOUR_APPID_HERE"),
    "apikey": os.getenv("XFYUN_API_KEY", "YOUR_API_KEY_HERE"),
    "apisecret": os.getenv("XFYUN_API_SECRET", "YOUR_API_SECRET_HERE"),
    "max_duration": 10.0,        # 最大录音时长(秒)
    "interval": 0.04,            # 推流间隔(秒)
    "sample_rate": 16000,        # 采样率(Hz)
    "language": "zh_cn",         # 语言
    "accent": "mandarin"         # 口音
}

# LLM配置 - API密钥从环境变量读取
# 设置方法:
#   Linux/Mac: export DOUBAO_API_KEY="your-api-key"
#   Windows: set DOUBAO_API_KEY=your-api-key
LLM_CONFIG = {
    "api_key": os.getenv("DOUBAO_API_KEY", "YOUR_API_KEY_HERE"),
    "model": "doubao-1.5-pro-32k-250115",
    "system_prompt": (
        "你现在是一个助老用户命令判断助手。你会收到老人提出的问题,其中包含老人要寻找的物品名称。"
        "你要做的是提取物品名称。注意,老人的提问可能包含一个或多个物品名称。"
        "你的回答只能为以下格式:[物品名称1, 物品名称2, ...]。"
        "如果你无法从问题中提取任何物品名称,请回答:[]。"
    ),
    "temperature": 0.7,
    "max_tokens": 100
}

# 视觉检测
VISION_CONFIG = {
    "img_size": 640,             # 推理图像尺寸
    "conf_threshold": 0.5,       # 置信度阈值
    "iou_threshold": 0.65,       # NMS IoU阈值
    "max_detections": 300,       # 最大检测数
    "device": "Ascend",          # 设备类型
    "device_id": 0               # 设备ID
}

# ==================== 业务配置 ====================
# 中英文物品映射表
OBJECT_MAPPING = {
    # 餐具类
    "水杯": "cup",
    "杯子": "cup",
    "碗": "bowl",
    "勺子": "spoon",
    "叉子": "fork",
    "刀": "knife",
    "筷子": "fork",  # COCO无筷子,暂用fork
    "盘子": "bowl",
    
    # 水果类
    "苹果": "apple",
    "香蕉": "banana",
    "橙子": "orange",
    "橘子": "orange",
    
    # 瓶罐类
    "瓶子": "bottle",
    
    # 电子产品
    "手机": "cell phone",
    "电话": "cell phone",
    "鼠标": "mouse",
    "键盘": "keyboard",
    "遥控器": "remote",
    "笔记本": "laptop",
    "笔记本电脑": "laptop",
    "电脑": "laptop",
    
    # 文具类
    "书": "book",
    "剪刀": "scissors",
    
    # 日用品
    "牙刷": "toothbrush",
    "吹风机": "hair drier",
    
    # 可扩展...
}

# 坐标映射参数
COORDINATE_MAPPING_CONFIG = {
    "pixel_to_meter_x": 4000,    # X轴像素到米的转换系数
    "pixel_to_meter_y": 3000,    # Y轴像素到米的转换系数
    "y_scale": 0.8,              # Y轴缩放系数
    "y_bias": 0.19,              # Y轴偏置
    "image_center_x": 320,       # 图像中心X坐标
    "image_center_y": 480        # 图像中心Y坐标(注意:Y轴从下往上)
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    "level": "INFO",             # 日志级别: DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s [%(levelname)s] %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "save_to_file": False,       # 是否保存到文件
    "log_dir": ROBOCODE_ROOT / "logs"
}

# ==================== 系统行为配置 ====================
SYSTEM_BEHAVIOR = {
    "enable_voice": True,        # 启用语音识别
    "enable_llm": True,          # 启用LLM解析
    "enable_vision": True,       # 启用视觉检测
    "enable_robot": True,        # 启用机械臂控制
    
    "show_detection_image": True,  # 显示检测结果图像
    "detection_display_time": 2000,  # 检测结果显示时间(ms)
    
    "retry_on_failure": True,    # 失败时重试
    "max_retries": 3,            # 最大重试次数
    
    "buzzer_enabled": True,      # 启用蜂鸣器提示
    "safety_check": True         # 启用安全检查
}

# ==================== 性能优化配置 ====================
PERFORMANCE_CONFIG = {
    "warmup_iterations": 3,      # 模型预热次数
    "use_async_inference": False,  # 使用异步推理(暂不支持)
    "enable_cache": True,        # 启用结果缓存
    "parallel_processing": False  # 并行处理(暂不支持)
}

# ==================== 调试配置 ====================
DEBUG_CONFIG = {
    "verbose": False,            # 详细输出
    "save_images": False,        # 保存检测图像
    "save_dir": ROBOCODE_ROOT / "debug_output",
    "print_timing": True,        # 打印耗时统计
    "simulate_robot": False      # 模拟机械臂(不发送真实指令)
}


# ==================== 配置验证 ====================
def validate_config():
    """验证配置有效性"""
    errors = []
    
    # 检查模型文件
    if not YOLOV8_MODEL_PATH.exists():
        errors.append(f"模型文件不存在: {YOLOV8_MODEL_PATH}")
    
    # 检查offset配置
    if not OFFSET_CONFIG_PATH.exists():
        errors.append(f"Offset配置文件不存在: {OFFSET_CONFIG_PATH}")
    
    # 检查摄像头ID
    if CAMERA_CONFIG["camera_id"] < 0:
        errors.append(f"无效的摄像头ID: {CAMERA_CONFIG['camera_id']}")
    
    # 检查阈值范围
    if not (0 < VISION_CONFIG["conf_threshold"] < 1):
        errors.append(f"置信度阈值超出范围: {VISION_CONFIG['conf_threshold']}")
    
    if errors:
        print("❌ 配置验证失败:")
        for err in errors:
            print(f"   - {err}")
        return False
    
    print("✅ 配置验证通过")
    return True


# ==================== 配置导出 ====================
def get_full_config():
    """获取完整配置字典"""
    return {
        "paths": {
            "model_path": str(YOLOV8_MODEL_PATH),
            "offset_path": str(OFFSET_CONFIG_PATH),
            "dp_bin_path": str(DP_BIN_PATH) if DP_BIN_PATH.exists() else None
        },
        "camera": CAMERA_CONFIG,
        "robot": ROBOT_ARM_CONFIG,
        "asr": ASR_CONFIG,
        "llm": LLM_CONFIG,
        "vision": VISION_CONFIG,
        "mapping": OBJECT_MAPPING,
        "coordinate": COORDINATE_MAPPING_CONFIG,
        "log": LOG_CONFIG,
        "behavior": SYSTEM_BEHAVIOR,
        "performance": PERFORMANCE_CONFIG,
        "debug": DEBUG_CONFIG
    }


def print_config_summary():
    """打印配置摘要"""
    print("="*60)
    print("系统配置摘要")
    print("="*60)
    print(f"模型路径: {YOLOV8_MODEL_PATH}")
    print(f"Offset配置: {OFFSET_CONFIG_PATH}")
    print(f"摄像头: ID={CAMERA_CONFIG['camera_id']}, {CAMERA_CONFIG['width']}x{CAMERA_CONFIG['height']}")
    print(f"检测阈值: conf={VISION_CONFIG['conf_threshold']}, iou={VISION_CONFIG['iou_threshold']}")
    print(f"物品映射: {len(OBJECT_MAPPING)} 个类别")
    print(f"启用模块: 语音={SYSTEM_BEHAVIOR['enable_voice']}, "
          f"LLM={SYSTEM_BEHAVIOR['enable_llm']}, "
          f"视觉={SYSTEM_BEHAVIOR['enable_vision']}, "
          f"机械臂={SYSTEM_BEHAVIOR['enable_robot']}")
    print("="*60)


if __name__ == "__main__":
    print_config_summary()
    validate_config()
