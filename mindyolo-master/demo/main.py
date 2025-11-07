import subprocess
import re
import ast
import LLM意图识别 as LLM
import test_voice as RV
import mindspore as ms
# ========== 可配置参数 ==========
IMAGE_PATH = "/home/HwHiAiUser/mindyolo-master/image17112/test.jpg"
CONFIG_PATH = "/home/HwHiAiUser/mindyolo-master/configs/yolov8/yolov8s.yaml"
WEIGHT_PATH = "/home/HwHiAiUser/mindyolo-master/yolov8s.ckpt"
DEVICE_PATH = "/dev/video0"
RESOLUTION = "1280x720"

# ========== COCO 类别映射表（ID→名称） ==========
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# 反向映射：名称→ID
COCO_NAME_TO_ID = {name: i for i, name in enumerate(COCO_NAMES)}

# ========== 1️⃣ 拍照 ==========
def capture_image():
    """调用 fswebcam 拍照"""
    cmd = ["fswebcam", "-d", DEVICE_PATH, "-r", RESOLUTION, IMAGE_PATH]
    print(f"[INFO] 拍照命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ 拍照失败: {result.stderr}")
    print("[INFO] 拍照完成 ✅")

# ========== 2️⃣ 调用预测脚本 ==========
def run_prediction():
    """调用 MindYOLO 的 predict.py 进行推理"""
    cmd = [
        "python3", "/home/HwHiAiUser/mindyolo-master/predict_1.py",
        "--config", CONFIG_PATH,
        "--weight", WEIGHT_PATH,
        "--image_path", IMAGE_PATH,
    ]
    print(f"[INFO] 推理命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("❌ 推理执行失败")
    print("[INFO] 推理完成 ✅")
    return result.stdout

# ========== 3️⃣ 解析输出 ==========
def parse_labels(output_text):
    """从 predict.py 输出中提取识别标签"""
    pattern = r"Predict result is:\s*(\{.*\})"
    match = re.search(pattern, output_text)
    if not match:
        raise ValueError("未在输出中找到识别结果。")

    result_dict = ast.literal_eval(match.group(1))
    if not result_dict["category_id"]:
        print("⚠️ 未检测到任何物体。")
        return {}

    detected = {}
    for cid, bbox, score in zip(result_dict["category_id"], result_dict["bbox"], result_dict["score"]):
        name = COCO_NAMES[cid]
        detected[name] = {"id": cid, "bbox": bbox, "score": score}

    print("\n✅ 识别结果:")
    for k, v in detected.items():
        print(f" - {k:<15} (score={v['score']:.2f})")
        print(detected)

    return detected

# ========== 4️⃣ 综合检测流程 ==========
def detect():
    try:
        capture_image()
        output = run_prediction()
        detected_objects = parse_labels(output)
        print("[INFO] 识别流程完成 ✅")
        return detected_objects
    except Exception as e:
        print(f"[ERROR] {e}")
        return {}

# ========== 5️⃣ 主程序逻辑 ==========
def main():
    print("[设备类型]", ms.context.get_context("device_target"))
    print("[模式]", ms.context.get_context("mode"))
    detected_objects = detect()  # { 'dog': {...}, 'bottle': {...} }
    print('识别完成，等待语音指令...')

    voice_text = RV.asr_recognize()  # 语音识别结果文本
    print(f"[🎤 语音识别结果]: {voice_text}")

    target = LLM.target_objects(voice_text)  # 提取语义中的目标对象名，如 "dog"
    print(f"[🤖 模型识别出的目标物体]: {target}")

    print(LLM.evaluate_targe_object(target, detected_objects))

if __name__ == "__main__":
    main()
