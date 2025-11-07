#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版测试脚本 - 不依赖机械臂,仅测试视觉检测+语音+LLM匹配
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "mindyolo-master" / "demo"))
sys.path.insert(0, str(Path(__file__).parent / "mindyolo-master"))


def test_voice_recognition():
    """测试语音识别模块"""
    logger.info("="*60)
    logger.info("测试1: 语音识别")
    logger.info("="*60)
    
    try:
        from recognize_voice import asr_recognize
        
        print("\n请说话(5秒)...")
        text = asr_recognize(max_duration=5.0, interval_sec=0.04)
        logger.info(f"✅ 识别结果: {text}")
        return text
    except Exception as e:
        logger.error(f"❌ 语音识别失败: {e}")
        return None


def test_llm_parsing(text):
    """测试LLM语义解析"""
    logger.info("="*60)
    logger.info("测试2: LLM语义解析")
    logger.info("="*60)
    
    try:
        from LLM意图识别 import target_objects
        
        targets = target_objects(text)
        logger.info(f"✅ 提取目标: {targets}")
        return targets
    except Exception as e:
        logger.error(f"❌ LLM解析失败: {e}")
        return None


def test_vision_detection():
    """测试视觉检测模块"""
    logger.info("="*60)
    logger.info("测试3: 视觉检测(使用predict_1.py)")
    logger.info("="*60)
    
    try:
        import mindspore as ms
        from mindspore import Tensor
        from mindyolo.utils.metrics import non_max_suppression, scale_coords
        
        # 设置环境
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
        
        # 加载模型
        model_path = r"d:\robocode\mindyolo-master\yolov8s_coco.mindir"
        logger.info(f"加载模型: {model_path}")
        
        graph = ms.load_mindir(model_path)
        network = ms.nn.GraphCell(graph)
        
        # 预热
        dummy = Tensor(np.ones((1, 3, 640, 640)), ms.float32)
        _ = network(dummy)
        logger.info("✅ 模型预热完成")
        
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error("❌ 摄像头读取失败")
            return None
        
        # 预处理
        img = cv2.resize(frame, (640, 480))
        h_ori, w_ori = img.shape[:2]
        
        img_input = cv2.resize(img, (640, 640))
        img_tensor = img_input[:, :, ::-1].transpose(2, 0, 1) / 255.0
        img_tensor = Tensor(img_tensor[None], ms.float32)
        
        # 推理
        t0 = time.time()
        out = network(img_tensor)
        infer_time = time.time() - t0
        
        # NMS
        out = out.asnumpy()
        out = non_max_suppression(out, conf_thres=0.5, iou_thres=0.65, need_nms=True)
        
        logger.info(f"⏱️ 推理耗时: {infer_time*1000:.1f}ms")
        
        # 解析结果
        class_names = [
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
        
        detections = []
        for pred in out:
            if len(pred) == 0:
                continue
            
            predn = np.copy(pred)
            scale_coords(img_tensor.shape[2:], predn[:, :4], (h_ori, w_ori))
            
            for det in predn:
                x1, y1, x2, y2, conf, cls_id = det
                cls_id = int(cls_id)
                
                detections.append({
                    "class_name": class_names[cls_id] if cls_id < len(class_names) else "unknown",
                    "confidence": float(conf),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
        
        logger.info(f"✅ 检测到 {len(detections)} 个物体: {[d['class_name'] for d in detections]}")
        
        # 可视化
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{det['class_name']} {det['confidence']:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Detection", img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        
        return detections
        
    except Exception as e:
        logger.error(f"❌ 视觉检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_target_matching(voice_target, detections):
    """测试目标匹配"""
    logger.info("="*60)
    logger.info("测试4: 目标匹配")
    logger.info("="*60)
    
    # 中英文映射
    mapping = {
        "水杯": "cup", "杯子": "cup",
        "苹果": "apple", "香蕉": "banana",
        "瓶子": "bottle", "碗": "bowl",
        "书": "book", "手机": "cell phone",
        "鼠标": "mouse", "键盘": "keyboard"
    }
    
    english_target = mapping.get(voice_target)
    if not english_target:
        logger.warning(f"⚠️ 未找到'{voice_target}'的映射")
        return None
    
    logger.info(f"🔍 映射: {voice_target} → {english_target}")
    
    candidates = [d for d in detections if d["class_name"] == english_target]
    
    if not candidates:
        logger.warning(f"❌ 未检测到: {english_target}")
        return None
    
    best = max(candidates, key=lambda x: x["confidence"])
    logger.info(f"✅ 匹配成功: {best['class_name']} (置信度: {best['confidence']:.2%})")
    
    return best


def main():
    """主测试流程"""
    logger.info("\n" + "="*60)
    logger.info("🧪 语音引导机械臂系统 - 集成测试")
    logger.info("="*60 + "\n")
    
    # 测试1: 语音识别
    voice_text = test_voice_recognition()
    if not voice_text:
        voice_text = "帮我拿水杯"  # 使用默认测试文本
        logger.info(f"使用默认测试文本: {voice_text}")
    
    time.sleep(1)
    
    # 测试2: LLM解析
    targets = test_llm_parsing(voice_text)
    if not targets:
        logger.error("❌ 测试中断: LLM解析失败")
        return
    
    target_name = targets[0]
    time.sleep(1)
    
    # 测试3: 视觉检测
    detections = test_vision_detection()
    if not detections:
        logger.error("❌ 测试中断: 视觉检测失败")
        return
    
    time.sleep(1)
    
    # 测试4: 目标匹配
    matched = test_target_matching(target_name, detections)
    
    if matched:
        logger.info("\n" + "="*60)
        logger.info("✅ 全流程测试成功!")
        logger.info(f"   语音输入: {voice_text}")
        logger.info(f"   目标提取: {target_name}")
        logger.info(f"   匹配结果: {matched['class_name']} (置信度: {matched['confidence']:.2%})")
        logger.info(f"   边界框: {matched['bbox']}")
        logger.info("="*60)
    else:
        logger.warning("\n⚠️ 未找到匹配目标")


if __name__ == "__main__":
    main()
