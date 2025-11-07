#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉引导机械臂抓取系统 - 主控程序
整合: 语音识别 → LLM语义解析 → YOLOv8视觉感知 → 目标匹配 → 机械臂执行
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# ==================== 配置日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 导入深度学习框架 - 优先NPU,自动降级CPU
try:
    import mindspore as ms
    from mindspore import Tensor
    MINDSPORE_AVAILABLE = True
    logger.info("✅ MindSpore已安装 (支持NPU推理)")
except ImportError:
    MINDSPORE_AVAILABLE = False
    logger.warning("⚠️ MindSpore未安装,将使用PyTorch (CPU模式)")

try:
    import torch
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch已安装 (CPU/GPU备用)")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch未安装")

# 导入子模块
sys.path.insert(0, str(Path(__file__).parent / "mindyolo-master" / "demo"))
sys.path.insert(0, str(Path(__file__).parent / "mindyolo-master"))

from mindyolo.models import create_model
from mindyolo.utils.config import parse_args
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh

# 导入本地模块
from mindyolo.demo.recognize_voice import asr_recognize
from mindyolo.demo.LLM意图识别 import target_objects

# ROS2相关导入
try:
    import rclpy
    from dofbot_info.srv import Kinemarics
    import Arm_Lib
    ROS2_AVAILABLE = True
    logger.info("✅ ROS2模块已安装 (支持机械臂控制)")
except ImportError:
    ROS2_AVAILABLE = False
    logger.warning("⚠️ ROS2模块未安装,机械臂功能将被禁用")


# ==================== COCO类别名称 ====================
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# ==================== 中英文物品映射表 ====================
# 目标物品: 苹果、橘子、杯子、瓶子
OBJECT_MAPPING = {
    "苹果": "apple",
    "橘子": "orange",
    "橙子": "orange",
    "杯子": "cup",
    "水杯": "cup",
    "瓶子": "bottle",
}

# ==================== 分拣位置配置 ====================
# 定义4个分拣区域的放置位置(关节角度)
SORTING_POSITIONS = {
    "apple": [45, 50, 20, 60, 265],    # 苹果 - 位置1(左前)
    "orange": [27, 75, 0, 50, 265],    # 橘子 - 位置2(左后)
    "cup": [147, 75, 0, 50, 265],      # 杯子 - 位置3(右后)
    "bottle": [133, 50, 20, 60, 265]   # 瓶子 - 位置4(右前)
}


# ==================== 视觉感知模块 ====================
class VisionPerception:
    """视觉感知模块 - 优先NPU,自动降级CPU"""
    
    def __init__(self, model_path_mindir: str = None, model_path_pt: str = None, 
                 config_path: str = None, img_size: int = 640, device: str = "auto"):
        """
        初始化视觉感知模块 - 智能选择最优推理后端
        
        Args:
            model_path_mindir: MindIR模型路径(.mindir) - 用于NPU
            model_path_pt: PyTorch模型路径(.pt) - 用于CPU/GPU
            config_path: 配置文件路径
            img_size: 推理图像尺寸
            device: 设备类型("auto", "npu", "cpu", "cuda")
        """
        self.img_size = img_size
        self.conf_thres = 0.5  # 置信度阈值
        self.iou_thres = 0.65   # NMS IoU阈值
        self.use_mindspore = False
        self.use_torch = False
        self.backend_name = "Unknown"
        
        logger.info("="*60)
        logger.info("🔄 初始化视觉感知模块")
        logger.info("="*60)
        
        # 策略: 优先NPU → GPU → CPU
        if device == "auto":
            device = self._auto_select_device()
        
        logger.info(f"📍 目标设备: {device}")
        
        # 1️⃣ 优先尝试NPU (MindSpore + 昇腾310B)
        if device == "npu" and MINDSPORE_AVAILABLE:
            if self._try_load_npu(model_path_mindir, img_size):
                return
        
        # 2️⃣ 降级: GPU (PyTorch + CUDA)
        if device == "cuda" and TORCH_AVAILABLE:
            if self._try_load_gpu(model_path_pt):
                return
        
        # 3️⃣ 最后降级: CPU (PyTorch)
        if TORCH_AVAILABLE:
            if self._try_load_cpu(model_path_pt):
                return
        
        # 4️⃣ 都失败则报错
        raise RuntimeError(
            "❌ 无可用推理后端!\n"
            "请安装以下之一:\n"
            "  - NPU: pip install mindspore (需昇腾驱动)\n"
            "  - CPU/GPU: pip install ultralytics torch"
        )
    
    def _auto_select_device(self) -> str:
        """自动选择最优设备"""
        # 检测NPU
        if MINDSPORE_AVAILABLE:
            try:
                import subprocess
                result = subprocess.run(['npu-smi', 'info'], 
                                       capture_output=True, timeout=2)
                if result.returncode == 0:
                    logger.info("✅ 检测到昇腾NPU")
                    return "npu"
            except:
                pass
        
        # 检测GPU
        if TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"✅ 检测到CUDA GPU: {torch.cuda.get_device_name(0)}")
                    return "cuda"
            except:
                pass
        
        # 默认CPU
        logger.info("ℹ️ 使用CPU模式")
        return "cpu"
    
    def _try_load_npu(self, model_path: str, img_size: int) -> bool:
        """尝试加载NPU模型"""
        try:
            logger.info("🚀 尝试加载NPU模型...")
            
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"⚠️ NPU模型文件不存在: {model_path}")
                return False
            
            # 设置Ascend上下文
            ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
            ms.set_recursion_limit(2000)
            
            # 加载MindIR模型
            graph = ms.load_mindir(model_path)
            self.network = ms.nn.GraphCell(graph)
            
            # 预热编译
            dummy = Tensor(np.ones((1, 3, img_size, img_size)), ms.float32)
            _ = self.network(dummy)
            
            self.use_mindspore = True
            self.backend_name = "NPU (Ascend 310B)"
            logger.info("✅ NPU模型加载成功! (昇腾310B)")
            logger.info(f"   模型路径: {model_path}")
            logger.info(f"   预期推理速度: ~30ms/帧")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ NPU加载失败: {e}")
            logger.info("   将尝试降级到CPU模式...")
            return False
    
    def _try_load_gpu(self, model_path: str) -> bool:
        """尝试加载GPU模型"""
        try:
            logger.info("🚀 尝试加载GPU模型...")
            from ultralytics import YOLO
            import torch
            
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA不可用")
                return False
            
            # 加载YOLOv8模型
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                logger.info("   使用预训练模型 (自动下载yolov8s.pt)")
                self.model = YOLO('yolov8s.pt')
            
            self.model.to('cuda')
            
            self.use_torch = True
            self.backend_name = f"GPU ({torch.cuda.get_device_name(0)})"
            logger.info("✅ GPU模型加载成功!")
            logger.info(f"   预期推理速度: ~20-50ms/帧")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ GPU加载失败: {e}")
            return False
    
    def _try_load_cpu(self, model_path: str) -> bool:
        """尝试加载CPU模型"""
        try:
            logger.info("🚀 加载CPU模型...")
            from ultralytics import YOLO
            
            # 加载YOLOv8模型
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                logger.info("   使用预训练模型 (自动下载yolov8s.pt)")
                self.model = YOLO('yolov8s.pt')
            
            self.model.to('cpu')
            
            self.use_torch = True
            self.backend_name = "CPU"
            logger.info("✅ CPU模型加载成功!")
            logger.info("   预期推理速度: ~100-200ms/帧")
            return True
            
        except Exception as e:
            logger.error(f"❌ CPU加载失败: {e}")
            return False
    
    def detect(self, img: np.ndarray) -> Dict:
        """执行目标检测 - 自动选择后端"""
        if self.use_torch:
            return self._detect_torch(img)
        elif self.use_mindspore:
            return self._detect_mindspore(img)
        else:
            raise RuntimeError("无可用检测后端")
    
    def _detect_torch(self, img: np.ndarray) -> Dict:
        """
        PyTorch/Ultralytics YOLO检测
        
        Args:
            img: BGR格式的输入图像
            
        Returns:
            检测结果字典
        """
        t0 = time.time()
        
        # Ultralytics YOLO推理
        results = self.model(img, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        result = results[0]  # 第一张图
        
        infer_time = time.time() - t0
        logger.info(f"⏱️ 推理耗时: {infer_time*1000:.1f}ms ({self.backend_name})")
        
        # 解析结果
        detections = []
        boxes = result.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # 计算中心点
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            detections.append({
                "class_id": cls_id,
                "class_name": COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else "unknown",
                "confidence": conf,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": (cx, cy)
            })
        
        logger.info(f"👁️ 检测到 {len(detections)} 个物体")
        return {"detections": detections}
    
    def _detect_mindspore(self, img: np.ndarray) -> Dict:
        """
        MindSpore检测(备用)
        
        Args:
            img: BGR格式的输入图像
            
        Returns:
            检测结果字典
        """
        h_ori, w_ori = img.shape[:2]
        # 1. 图像预处理
        r = self.img_size / max(h_ori, w_ori)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img_resized = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
        else:
            img_resized = img.copy()
        
        # Padding到640x640
        h, w = img_resized.shape[:2]
        if h < self.img_size or w < self.img_size:
            dh = (self.img_size - h) / 2
            dw = (self.img_size - w) / 2
            img_resized = cv2.copyMakeBorder(
                img_resized, int(dh), int(dh), int(dw), int(dw),
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
        
        # 转换为Tensor (NCHW, RGB, [0,1])
        img_tensor = img_resized[:, :, ::-1].transpose(2, 0, 1) / 255.0
        img_tensor = Tensor(img_tensor[None], ms.float32)
        
        # 2. NPU推理
        t0 = time.time()
        out = self.network(img_tensor)
        infer_time = time.time() - t0
        
        # 3. NMS后处理
        out = out.asnumpy()
        t1 = time.time()
        out = non_max_suppression(out, conf_thres=self.conf_thres, iou_thres=self.iou_thres, need_nms=True)
        nms_time = time.time() - t1
        
        logger.info(f"⏱️ 推理耗时: {infer_time*1000:.1f}ms | NMS: {nms_time*1000:.1f}ms")
        
        # 4. 解析结果
        detections = []
        for pred in out:
            if len(pred) == 0:
                continue
            
            # 坐标映射回原图
            predn = np.copy(pred)
            scale_coords(img_tensor.shape[2:], predn[:, :4], (h_ori, w_ori))
            
            for det in predn:
                x1, y1, x2, y2, conf, cls_id = det
                cls_id = int(cls_id)
                
                # 计算中心点
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else "unknown",
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": (cx, cy)
                })
        
        logger.info(f"👁️ 检测到 {len(detections)} 个物体")
        return {"detections": detections}


# ==================== 决策层模块 ====================
class DecisionLayer:
    """决策层 - 负责目标匹配和任务生成"""
    
    @staticmethod
    def match_target(voice_target: str, visual_detections: List[Dict]) -> Optional[Dict]:
        """
        匹配语音目标与视觉检测结果
        
        Args:
            voice_target: 语音提取的中文目标(如"水杯")
            visual_detections: 视觉检测结果列表
            
        Returns:
            匹配成功返回最佳目标,失败返回None
        """
        # 1. 中英文映射
        english_target = OBJECT_MAPPING.get(voice_target)
        if not english_target:
            logger.warning(f"⚠️ 未找到'{voice_target}'的英文映射")
            return None
        
        logger.info(f"🔍 匹配目标: {voice_target} → {english_target}")
        
        # 2. 在检测结果中查找
        candidates = [
            det for det in visual_detections 
            if det["class_name"] == english_target
        ]
        
        if not candidates:
            logger.warning(f"❌ 未检测到目标物体: {english_target}")
            return None
        
        # 3. 选择最高置信度的目标
        best_match = max(candidates, key=lambda x: x["confidence"])
        logger.info(f"✅ 匹配成功: {best_match['class_name']} (置信度: {best_match['confidence']:.2%})")
        
        return best_match


# ==================== 坐标映射模块 ====================
class CoordinateMapper:
    """坐标映射模块 - 像素坐标转机械臂基坐标"""
    
    def __init__(self, offset_path: str, dp_bin_path: Optional[str] = None):
        """
        初始化坐标映射
        
        Args:
            offset_path: offset.txt文件路径
            dp_bin_path: 透视变换参数文件路径(可选)
        """
        # 读取偏移量
        with open(offset_path, 'r') as f:
            self.y_offset = float(f.readline().strip())
            self.x_offset = float(f.readline().strip())
        
        logger.info(f"📐 坐标偏移: x={self.x_offset}, y={self.y_offset}")
        
        self.dp_bin_path = dp_bin_path
    
    def pixel_to_robot_base(self, pixel_x: int, pixel_y: int, img_width: int = 640, img_height: int = 480) -> Tuple[float, float]:
        """
        像素坐标转机械臂基坐标
        
        Args:
            pixel_x, pixel_y: 像素坐标
            img_width, img_height: 图像尺寸
            
        Returns:
            (x, y) 机械臂基坐标系下的坐标
        """
        # 参考garbage_identify.py的转换公式
        a = round(((pixel_x - 320) / 4000), 5)
        b = round(((480 - pixel_y) / 3000) * 0.8 + 0.19, 5)
        
        # 应用偏移补偿
        x = a + self.x_offset
        y = b + self.y_offset
        
        logger.info(f"📍 坐标映射: 像素({pixel_x}, {pixel_y}) → 机械臂({x:.4f}, {y:.4f})")
        return (x, y)


# ==================== 机械臂执行模块 ====================
class RobotArmController:
    """机械臂执行模块 - ROS2服务调用 + 逆运动学"""
    
    def __init__(self):
        """初始化机械臂控制器"""
        if not ROS2_AVAILABLE:
            logger.error("❌ ROS2不可用,机械臂控制器初始化失败")
            return
        
        # 检查ROS2是否已初始化(参考garbage_identify.py line 30)
        if not rclpy.ok():
            rclpy.init()
            logger.info("🔧 ROS2初始化完成")
        
        self.node = rclpy.create_node("voice_robot_controller")
        self.client = self.node.create_client(Kinemarics, "trial_service")
        self.arm = Arm_Lib.Arm_Device()
        
        logger.info("✅ 机械臂控制器初始化完成")
    
    def inverse_kinematics(self, x: float, y: float, z: float = 0.0) -> List[float]:
        """
        调用ROS2逆运动学服务
        
        Args:
            x, y, z: 目标位姿(基坐标系)
            
        Returns:
            关节角度列表 [j1, j2, j3, j4, j5]
        """
        if not ROS2_AVAILABLE:
            return [90, 90, 0, 0, 90]
        
        self.client.wait_for_service(timeout_sec=2.0)
        
        request = Kinemarics.Request()
        request.tar_x = x
        request.tar_y = y
        request.tar_z = z
        request.kin_name = "ik"
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        
        response = future.result()
        if response:
            joints = [
                response.joint1,
                response.joint2,
                response.joint3,
                response.joint4,
                response.joint5
            ]
            
            # 角度调整(参考garbage_identify.py)
            if joints[2] < 0:
                joints[1] += joints[2] / 2
                joints[3] += joints[2] * 3 / 4
                joints[2] = 0
            
            logger.info(f"🤖 逆运动学解算: ({x:.3f}, {y:.3f}, {z:.3f}) → {joints}")
            return joints
        
        return None
    
    def grasp_and_place(self, joints: List[float], target_class: str, xy_init: List[int] = [90, 135]):
        """
        执行抓取+分拣动作
        
        Args:
            joints: 目标关节角度
            target_class: 目标类别("apple", "orange", "cup", "bottle")
            xy_init: 初始位置
        """
        if not ROS2_AVAILABLE:
            logger.warning("⚠️ ROS2不可用,跳过机械臂动作")
            return
        
        logger.info(f"🤖 开始执行抓取动作: {target_class}")
        
        # 蜂鸣器提示
        self.arm.Arm_Buzzer_On(1)
        time.sleep(0.5)
        
        grap_joint = 130  # 夹爪闭合角度
        
        # 1. 移动到目标上方
        joints_up = [joints[0], 80, 50, 50, 265, 30]
        self.arm.Arm_serial_servo_write6_array(joints_up, 1000)
        time.sleep(1)
        
        # 2. 松开夹爪
        self.arm.Arm_serial_servo_write(6, 0, 500)
        time.sleep(0.5)
        
        # 3. 移动到目标位置
        joints_target = [joints[0], joints[1], joints[2], joints[3], 265, 30]
        self.arm.Arm_serial_servo_write6_array(joints_target, 500)
        time.sleep(0.5)
        
        # 4. 夹紧
        self.arm.Arm_serial_servo_write(6, grap_joint, 500)
        time.sleep(0.5)
        
        # 5. 抬起
        self.arm.Arm_serial_servo_write6_array(joints_up, 1000)
        time.sleep(1)
        
        # 6. 移动到分拣位置
        if target_class in SORTING_POSITIONS:
            sorting_joints = SORTING_POSITIONS[target_class] + [grap_joint]
            logger.info(f"📦 移动到分拣位置: {target_class}")
            self.arm.Arm_serial_servo_write6_array(sorting_joints, 1000)
            time.sleep(1)
            
            # 7. 释放物体
            self.arm.Arm_serial_servo_write(6, 30, 500)
            time.sleep(0.5)
        else:
            logger.warning(f"⚠️ 未找到{target_class}的分拣位置,放回初始位置")
        
        # 8. 返回初始位置
        joints_init = [xy_init[0], xy_init[1], 0, 0, 90, 30]
        self.arm.Arm_serial_servo_write6_array(joints_init, 1000)
        time.sleep(1)
        
        logger.info("✅ 抓取动作完成")


# ==================== 主系统 ====================
class VoiceGuidedRobotSystem:
    """语音引导机械臂系统 - 主控类"""
    
    def __init__(self, config: Dict):
        """
        初始化系统
        
        Args:
            config: 配置字典 {
                "model_path": YOLOv8模型路径,
                "offset_path": offset.txt路径,
                "camera_id": 摄像头ID
            }
        """
        logger.info("="*60)
        logger.info("🚀 启动视觉引导机械臂抓取系统")
        logger.info("="*60)
        
        # 1. 初始化视觉模块 - 优先NPU
        model_path_mindir = config.get("model_path_mindir")  # NPU模型
        model_path_pt = config.get("model_path_pt")  # CPU/GPU模型
        device = config.get("device", "auto")  # auto/npu/cpu/cuda
        
        self.vision = VisionPerception(
            model_path_mindir=model_path_mindir,
            model_path_pt=model_path_pt,
            config_path=None,
            img_size=640,
            device=device
        )
        
        # 2. 初始化坐标映射
        self.mapper = CoordinateMapper(offset_path=config["offset_path"])
        
        # 3. 初始化机械臂控制器
        self.robot = RobotArmController() if ROS2_AVAILABLE else None
        
        # 4. 打开摄像头
        self.camera_id = config.get("camera_id", 0)
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 无法打开摄像头 {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        logger.info("✅ 系统初始化完成")
    
    def run_once(self, enable_voice=True, enable_llm=True):
        """执行一次完整的抓取流程"""
        logger.info("\n" + "="*60)
        logger.info("🎙️ 步骤1: 语音输入")
        logger.info("="*60)
        
        voice_text = ""
        target_name = ""
        
        # 1. 语音识别(可选)
        if enable_voice:
            print("\n▶️ 请说出您的指令(如: 帮我拿苹果)...")
            try:
                voice_text = asr_recognize(max_duration=5.0, interval_sec=0.04)
                logger.info(f"📝 识别结果: {voice_text}")
            except Exception as e:
                logger.error(f"❌ 语音识别失败: {e}")
                enable_llm = False
        
        # 2. LLM语义解析(可选)
        if enable_llm and voice_text:
            logger.info("\n" + "="*60)
            logger.info("🧠 步骤2: 语义解析")
            logger.info("="*60)
            
            try:
                target_list = target_objects(voice_text)
                if not target_list:
                    logger.warning("⚠️ 未识别到目标物品")
                    return False
                
                target_name = target_list[0]  # 取第一个目标
                logger.info(f"🎯 提取目标: {target_name}")
            except Exception as e:
                logger.error(f"❌ 语义解析失败: {e}")
                return False
        
        # 如果没有语音输入,直接检测所有4个类别
        if not target_name:
            logger.info("ℹ️ 无语音输入,将检测所有目标类别: 苹果/橘子/杯子/瓶子")
        
        # 3. 视觉感知
        logger.info("\n" + "="*60)
        logger.info("👁️ 步骤3: 视觉感知")
        logger.info("="*60)
        
        ret, frame = self.cap.read()
        if not ret:
            logger.error("❌ 摄像头读取失败")
            return False
        
        # 调整图像尺寸
        frame = cv2.resize(frame, (640, 480))
        
        # 执行检测
        result = self.vision.detect(frame)
        detections = result["detections"]
        
        # 可视化
        vis_img = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{det['class_name']} {det['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cx, cy = det["center"]
            cv2.circle(vis_img, (cx, cy), 5, (0, 0, 255), -1)
        
        cv2.imshow("Detection Result", vis_img)
        cv2.waitKey(2000)  # 显示2秒
        
        # 4. 目标匹配
        logger.info("\n" + "="*60)
        logger.info("🔍 步骤4: 目标匹配")
        logger.info("="*60)
        
        matched_target = DecisionLayer.match_target(target_name, detections)
        if not matched_target:
            logger.warning(f"❌ 未找到目标物体: {target_name}")
            return False
        
        # 5. 坐标映射
        logger.info("\n" + "="*60)
        logger.info("📐 步骤5: 坐标映射")
        logger.info("="*60)
        
        cx, cy = matched_target["center"]
        robot_x, robot_y = self.mapper.pixel_to_robot_base(cx, cy)
        
        # 6. 逆运动学+执行
        logger.info("\n" + "="*60)
        logger.info("🤖 步骤6: 机械臂执行")
        logger.info("="*60)
        
        if self.robot:
            joints = self.robot.inverse_kinematics(robot_x, robot_y, z=0.0)
            if joints:
                # 传递target_class进行分拣
                english_target = OBJECT_MAPPING.get(target_name, matched_target["class_name"])
                self.robot.grasp_and_place(joints, target_class=english_target)
                logger.info("✅ 任务完成!")
                return True
        else:
            logger.warning("⚠️ 机械臂不可用,仅模拟坐标映射")
            logger.info(f"📍 目标坐标: ({robot_x:.4f}, {robot_y:.4f})")
        
        return True
    
    def run_continuous(self):
        """持续运行模式"""
        logger.info("\n🔁 进入持续运行模式 (按Ctrl+C退出)")
        
        try:
            while True:
                self.run_once()
                logger.info("\n⏳ 等待5秒后执行下一次...")
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("\n🛑 用户中断,系统退出")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("♻️ 资源已释放")


# ==================== 主入口 ====================
def main():
    # 配置参数 - 优先NPU,自动降级CPU
    import platform
    
    # 基础配置
    config = {
        "model_path_mindir": None,  # NPU模型(.mindir)
        "model_path_pt": None,      # CPU/GPU模型(.pt)
        "offset_path": "",
        "camera_id": 0,
        "device": "auto"  # auto: 自动选择(NPU>GPU>CPU)
    }
    
    # 根据操作系统设置路径
    if platform.system() == "Windows":
        # Windows环境 - 支持NPU/CPU
        config["model_path_mindir"] = r"d:\robocode\mindyolo-master\yolov8s_coco.mindir"  # NPU模型
        config["model_path_pt"] = r"d:\robocode\mindyolo-master\yolov8s.pt"  # CPU备用
        config["offset_path"] = r"d:\robocode\ros2_robot_arm\ros2_ws\src\dofbot_garbage_yolov5\dofbot_garbage_yolov5\config\offset.txt"
    else:  # Linux/Ubuntu
        # Ubuntu环境 - 仅CPU
        config["model_path_pt"] = "/home/user/robocode/mindyolo-master/yolov8s.pt"
        config["offset_path"] = "/home/user/robocode/ros2_robot_arm/ros2_ws/src/dofbot_garbage_yolov5/dofbot_garbage_yolov5/config/offset.txt"
    
    logger.info("\n" + "="*70)
    logger.info("🎯 系统启动信息")
    logger.info("="*70)
    logger.info(f"💻 操作系统: {platform.system()}")
    logger.info(f"📦 NPU模型: {config['model_path_mindir'] or '未配置'}")
    logger.info(f"📦 CPU模型: {config['model_path_pt'] or '自动下载'}")
    logger.info(f"🎯 设备模式: {config['device']} (优先NPU)")
    logger.info(f"🎯 目标物品: 苹果/橘子/杯子/瓶子")
    logger.info("="*70 + "\n")
    
    # 创建系统实例
    system = VoiceGuidedRobotSystem(config)
    
    # 运行模式选择
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["once", "continuous", "vision_only"], default="vision_only",
                       help="运行模式: once(单次) 或 continuous(持续) 或 vision_only(仅视觉)")
    parser.add_argument("--no-voice", action="store_true", help="禁用语音识别")
    parser.add_argument("--no-llm", action="store_true", help="禁用LLM解析")
    args = parser.parse_args()
    
    if args.mode == "once":
        system.run_once(enable_voice=not args.no_voice, enable_llm=not args.no_llm)
    elif args.mode == "vision_only":
        logger.info("👁️ 仅视觉检测模式 (无语音/LLM)")
        system.run_once(enable_voice=False, enable_llm=False)
    else:
        system.run_continuous()


if __name__ == "__main__":
    main()
