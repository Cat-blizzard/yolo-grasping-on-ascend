#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4类物品检测测试脚本 - CPU模式
目标: 苹果、橘子、杯子、瓶子
"""

import cv2
import time
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 中英文映射
OBJECT_MAPPING = {
    "苹果": "apple",
    "橘子": "orange",
    "橙子": "orange",
    "杯子": "cup",
    "水杯": "cup",
    "瓶子": "bottle",
}

# 分拣位置
SORTING_POSITIONS = {
    "apple": "位置1(左前)",
    "orange": "位置2(左后)",
    "cup": "位置3(右后)",
    "bottle": "位置4(右前)"
}


def test_yolo_detection():
    """测试YOLOv8检测4类物品"""
    logger.info("="*60)
    logger.info("4类物品检测测试 (CPU模式)")
    logger.info("目标: 苹果(apple), 橘子(orange), 杯子(cup), 瓶子(bottle)")
    logger.info("="*60)
    
    try:
        from ultralytics import YOLO
        logger.info("✓ Ultralytics YOLOv8 已安装")
    except ImportError:
        logger.error("✗ Ultralytics未安装, 请运行: pip install ultralytics")
        return False
    
    # 加载模型
    logger.info("\n正在加载YOLOv8s模型 (首次运行会自动下载~22MB)...")
    model = YOLO('yolov8s.pt')
    logger.info("✓ 模型加载完成")
    
    # 打开摄像头
    logger.info("\n正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("✗ 无法打开摄像头!")
        logger.info("提示: 检查摄像头连接或权限 (ls /dev/video*)")
        return False
    
    logger.info("✓ 摄像头已打开")
    logger.info("\n=" * 60)
    logger.info("开始实时检测 (按 'q' 退出)")
    logger.info("=" * 60)
    
    # 目标类别 (COCO索引)
    target_classes = {
        47: "apple",    # 苹果
        49: "orange",   # 橘子
        41: "cup",      # 杯子
        39: "bottle"    # 瓶子
    }
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ 无法读取摄像头帧")
                break
            
            frame_count += 1
            
            # 每10帧执行一次检测(降低CPU负载)
            if frame_count % 10 == 0:
                # YOLOv8推理
                t0 = time.time()
                results = model(frame, conf=0.4, iou=0.5, verbose=False)
                infer_time = time.time() - t0
                
                result = results[0]
                
                # 统计检测结果
                detected_objects = {}
                
                # 绘制检测框
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    
                    # 只处理4类目标
                    if cls_id in target_classes:
                        class_name = target_classes[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 绘制边界框
                        color = {
                            "apple": (0, 255, 0),    # 绿色
                            "orange": (0, 165, 255), # 橙色
                            "cup": (255, 0, 0),      # 蓝色
                            "bottle": (255, 255, 0)  # 青色
                        }.get(class_name, (128, 128, 128))
                        
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                     color, 2)
                        
                        # 添加标签
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # 统计
                        if class_name not in detected_objects:
                            detected_objects[class_name] = []
                        detected_objects[class_name].append(conf)
                
                # 打印检测结果
                if detected_objects:
                    logger.info(f"\n[帧{frame_count}] 推理: {infer_time*1000:.1f}ms")
                    for obj_name, confs in detected_objects.items():
                        chinese_name = [k for k, v in OBJECT_MAPPING.items() if v == obj_name][0]
                        sorting_pos = SORTING_POSITIONS[obj_name]
                        logger.info(f"  ✓ {chinese_name}({obj_name}): {len(confs)}个 "
                                   f"(置信度: {max(confs):.2%}) → 分拣至{sorting_pos}")
            
            # 显示FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # 显示提示
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示图像
            cv2.imshow("4-Class Object Detection", frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ 用户中断")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # 统计信息
        total_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("统计信息")
        logger.info("="*60)
        logger.info(f"总帧数: {frame_count}")
        logger.info(f"运行时间: {total_time:.1f}秒")
        logger.info(f"平均FPS: {frame_count/total_time:.1f}")
        logger.info("="*60)
    
    return True


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  4类物品检测测试脚本")
    print("  目标: 苹果、橘子、杯子、瓶子")
    print("  模式: CPU实时检测")
    print("="*60 + "\n")
    
    # 测试依赖
    try:
        import cv2
        import numpy
        logger.info("✓ 依赖检查通过")
    except ImportError as e:
        logger.error(f"✗ 缺少依赖: {e}")
        logger.info("请运行: pip install opencv-python numpy ultralytics")
        return
    
    # 执行测试
    success = test_yolo_detection()
    
    if success:
        logger.info("\n✅ 测试完成!")
        logger.info("\n提示:")
        logger.info("  1. 将苹果/橘子/杯子/瓶子放在摄像头前测试")
        logger.info("  2. 观察检测框颜色: 苹果(绿), 橘子(橙), 杯子(蓝), 瓶子(青)")
        logger.info("  3. 确认分拣位置提示是否正确")
    else:
        logger.error("\n❌ 测试失败!")


if __name__ == "__main__":
    main()
